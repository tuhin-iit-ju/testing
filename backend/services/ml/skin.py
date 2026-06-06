"""
Skin disease inference service.
Matches the architecture detection logic from skin disease/skin_model_package/model1.py.
"""
import base64
import io
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from config import settings

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_INFO = {
    "akiec": "Actinic Keratoses / Intraepithelial Carcinoma",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis-like Lesions",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevi",
    "vasc":  "Vascular Lesions",
}
NUM_CLASSES = len(CLASS_NAMES)

# Per-variant input sizes used during the original training runs.
# Used to build the right transform for each model in the ensemble.
_TIMM_SIZES = {"b0": 224, "b1": 240, "b2": 260, "b3": 300, "b4": 380}


def _make_transform(size: int):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


_models: dict | None = None     # name -> {"model": ..., "transform": ...}
_device: torch.device | None = None


def _detect_arch(state_dict: dict) -> tuple[str, str | None]:
    """
    Returns (family, variant) where:
      family  ∈ {"timm_efficientnet", "torchvision_efficientnet", "resnet50"}
      variant ∈ {"b0","b1","b2","b3","b4"} for efficientnet families, else None
    """
    keys = list(state_dict.keys())
    if "conv_stem.weight" in keys:
        # timm efficientnet — distinguish B0/B1/B3/... by counting blocks per stage
        return "timm_efficientnet", _detect_timm_efficientnet_variant(state_dict)
    if any("features" in k for k in keys):
        return "torchvision_efficientnet", "b1"
    return "resnet50", None


def _detect_timm_efficientnet_variant(state_dict: dict) -> str:
    """
    timm's EfficientNet keys look like `blocks.<stage>.<idx>.<sub>`.
    Counting the max idx per stage gives a fingerprint of the variant:
      B0: blocks per stage = [1, 2, 2, 3, 3, 4, 1]
      B1: [2, 3, 3, 4, 4, 5, 2]
      B2: [2, 3, 3, 4, 4, 5, 2]  (same depth as B1, wider channels)
      B3: [2, 3, 3, 5, 5, 6, 2]
      B4: [2, 4, 4, 6, 6, 8, 2]
    """
    max_idx = {}
    for k in state_dict.keys():
        parts = k.split(".")
        if len(parts) >= 3 and parts[0] == "blocks" and parts[1].isdigit() and parts[2].isdigit():
            stage = int(parts[1])
            idx   = int(parts[2])
            max_idx[stage] = max(max_idx.get(stage, -1), idx)

    counts = [max_idx.get(i, -1) + 1 for i in range(7)]
    if counts[1] >= 4:
        return "b4"
    if counts[3] >= 5 or counts[5] >= 6:
        return "b3"
    if counts[3] >= 4 or counts[1] >= 2:
        return "b1"
    return "b0"


def _build(family: str, variant: str | None):
    if family == "timm_efficientnet":
        import timm
        return timm.create_model(f"efficientnet_{variant}", num_classes=NUM_CLASSES, pretrained=False)
    if family == "torchvision_efficientnet":
        m = models.efficientnet_b1(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
        return m
    m = models.resnet50(weights=None)
    m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
    return m


def _load_single(path: str, device: torch.device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    family, variant = _detect_arch(state)

    # Key normalization for common mismatches
    if family == "resnet50":
        state = {k.replace("fc.1.", "fc."): v for k, v in state.items()}
    if family == "timm_efficientnet":
        state = {k.replace("classifier.1.", "classifier."): v for k, v in state.items()}

    m = _build(family, variant)
    m.load_state_dict(state)
    m.to(device)
    m.eval()

    # Pick the right input transform for this architecture
    if family == "timm_efficientnet":
        size = _TIMM_SIZES.get(variant, 224)
    elif family == "torchvision_efficientnet":
        size = 240
    else:
        size = 224

    tag = f"{family}/{variant}" if variant else family
    return {"model": m, "transform": _make_transform(size), "arch": tag}


def _get_models():
    global _models, _device
    if _models is not None:
        return _models, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _models = {}
    for name, path in [
        ("model1", settings.SKIN_MODEL1_PATH),
        ("model2", settings.SKIN_MODEL2_PATH),
        ("model3", settings.SKIN_MODEL3_PATH),
    ]:
        if not os.path.exists(path):
            print(f"[SKIN] Skipping {name} — not found: {path}")
            continue
        try:
            _models[name] = _load_single(path, _device)
            print(f"[SKIN] Loaded {name}  ({_models[name]['arch']})")
        except Exception as e:
            print(f"[SKIN] Failed to load {name}: {e}")

    return _models, _device


def _gradcam_resnet(model, tensor: torch.Tensor, class_idx: int) -> Optional[str]:
    try:
        activations, gradients = [], []

        def fwd(_, __, out): activations.append(out)
        def bwd(_, __, g): gradients.append(g[0])

        h1 = model.layer4.register_forward_hook(fwd)
        h2 = model.layer4.register_full_backward_hook(bwd)

        tensor.requires_grad_(True)
        out = model(tensor)
        model.zero_grad()
        out[0, class_idx].backward()
        h1.remove(); h2.remove()

        acts = activations[0].detach().cpu().numpy()[0]
        grads = gradients[0].detach().cpu().numpy()[0]
        weights = grads.mean(axis=(1, 2))
        cam = (weights[:, None, None] * acts).sum(axis=0)
        cam = np.maximum(cam, 0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        import cv2
        cam_r = cv2.resize(cam, (224, 224))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_r), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        orig = tensor[0].detach().cpu().numpy().transpose(1, 2, 0)
        orig = np.clip((orig * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255, 0, 255).astype(np.uint8)
        blended = (0.4 * heatmap + 0.6 * orig).astype(np.uint8)

        buf = io.BytesIO()
        Image.fromarray(blended).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print(f"[SKIN] Grad-CAM failed: {e}")
        return None


def _predict_single_tta(model, batch: torch.Tensor) -> tuple[np.ndarray, dict]:
    """Run a model on (orig, hflip) batch and return mean probs + per-model summary dict."""
    with torch.inference_mode():
        probs = torch.softmax(model(batch), dim=1).cpu().numpy()
    mean_probs = probs.mean(axis=0)
    top_idx = int(np.argmax(mean_probs))
    summary = {
        "predicted_class": CLASS_NAMES[top_idx],
        "confidence": float(mean_probs[top_idx]),
        "all_probabilities": {CLASS_NAMES[i]: round(float(mean_probs[i]), 4) for i in range(NUM_CLASSES)},
    }
    return mean_probs, summary


def predict(image_bytes: bytes) -> dict:
    loaded, device = _get_models()
    if not loaded:
        raise RuntimeError("No skin models loaded")

    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    per_model_probs = []
    results = {}
    # Cache transform-keyed batches so models sharing a transform reuse the tensor.
    batch_cache: dict[int, torch.Tensor] = {}

    for name, bundle in loaded.items():
        model     = bundle["model"]
        transform = bundle["transform"]

        # Identify the transform's output size via its first child (Resize)
        # so we can dedupe equivalent batches.
        size_key = transform.transforms[0].size
        size_key = size_key if isinstance(size_key, int) else tuple(size_key)
        if size_key not in batch_cache:
            t = transform(pil_image).unsqueeze(0).to(device)
            batch_cache[size_key] = torch.cat([t, torch.flip(t, dims=[3])], dim=0)

        batch = batch_cache[size_key]
        probs, summary = _predict_single_tta(model, batch)
        per_model_probs.append(probs)
        results[name] = summary

    ensemble = np.stack(per_model_probs, axis=0).mean(axis=0)
    sorted_idx = np.argsort(ensemble)[::-1]
    top_idx = int(sorted_idx[0])
    top_conf = float(ensemble[top_idx])
    margin = float(ensemble[sorted_idx[0]] - ensemble[sorted_idx[1]]) if len(sorted_idx) > 1 else 1.0
    inconclusive = top_conf < 0.50 or margin < 0.15
    prediction = CLASS_NAMES[top_idx]

    gradcam_b64 = None
    if "model1" in loaded:
        try:
            m1 = loaded["model1"]["model"]
            if hasattr(m1, "layer4"):
                gc_tensor = loaded["model1"]["transform"](pil_image).unsqueeze(0).to(device)
                gradcam_b64 = _gradcam_resnet(m1, gc_tensor, top_idx)
        except Exception as e:
            print(f"[SKIN] Grad-CAM skipped: {e}")

    return {
        "prediction": prediction,
        "description": CLASS_INFO.get(prediction, "Unknown"),
        "confidence": round(top_conf, 4),
        "margin": round(margin, 4),
        "inconclusive": inconclusive,
        "all_probabilities": {CLASS_NAMES[i]: round(float(ensemble[i]), 4) for i in range(NUM_CLASSES)},
        "model1_result": results.get("model1"),
        "model2_result": results.get("model2"),
        "model3_result": results.get("model3"),
        "winner_model": "ensemble",
        "gradcam_image": gradcam_b64,
    }
