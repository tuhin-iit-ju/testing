"""
Chest X-Ray inference service.
Matches the exact architecture from X-Ray/Test/test_xray.py.
"""
import base64
import io
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from config import settings

TARGET_COLS = ["No Finding", "Cardiomegaly", "Edema", "Consolidation", "Pneumonia", "Atelectasis"]
NUM_CLASSES = len(TARGET_COLS)
THRESHOLD = 0.5

_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_models: dict | None = None
_device: torch.device | None = None


def _build_densenet():
    m = models.densenet121(weights=None)
    m.classifier = nn.Sequential(nn.Linear(m.classifier.in_features, NUM_CLASSES), nn.Sigmoid())
    return m


def _build_resnet50():
    m = models.resnet50(weights=None)
    m.fc = nn.Sequential(nn.Linear(m.fc.in_features, NUM_CLASSES), nn.Sigmoid())
    return m


def _build_vit():
    m = models.vit_b_16(weights=None)
    m.heads.head = nn.Sequential(nn.Linear(m.heads.head.in_features, NUM_CLASSES), nn.Sigmoid())
    return m


_REGISTRY = [
    ("DenseNet121", settings.XRAY_DENSENET_PATH, _build_densenet),
    ("ResNet50",    settings.XRAY_RESNET_PATH,    _build_resnet50),
    ("ViT-Base",    settings.XRAY_VIT_PATH,        _build_vit),
]


def _get_models():
    global _models, _device
    if _models is not None:
        return _models, _device

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _models = {}
    import os
    for name, path, builder in _REGISTRY:
        if not os.path.exists(path):
            print(f"[XRAY] Skipping {name} — file not found: {path}")
            continue
        m = builder()
        m.load_state_dict(torch.load(path, map_location=_device))
        m.to(_device)
        m.eval()
        _models[name] = m
        print(f"[XRAY] Loaded {name}")

    return _models, _device


def _gradcam_densenet(model, tensor: torch.Tensor, class_idx: int) -> Optional[str]:
    """Simple Grad-CAM for DenseNet121. Returns base64-encoded PNG or None."""
    try:
        activations, gradients = [], []

        def fwd_hook(_, __, out): activations.append(out)
        def bwd_hook(_, __, grad_out): gradients.append(grad_out[0])

        target_layer = model.features.denseblock4
        h1 = target_layer.register_forward_hook(fwd_hook)
        h2 = target_layer.register_full_backward_hook(bwd_hook)

        tensor.requires_grad_(True)
        output = model(tensor)
        model.zero_grad()
        output[0, class_idx].backward()

        h1.remove(); h2.remove()

        acts = activations[0].detach().cpu().numpy()[0]      # (C, H, W)
        grads = gradients[0].detach().cpu().numpy()[0]        # (C, H, W)
        weights = grads.mean(axis=(1, 2))                     # (C,)
        cam = (weights[:, None, None] * acts).sum(axis=0)
        cam = np.maximum(cam, 0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        import cv2
        cam_resized = cv2.resize(cam, (224, 224))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Blend with original image
        orig_np = tensor[0].detach().cpu().numpy().transpose(1, 2, 0)
        orig_np = (orig_np * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
        orig_np = np.clip(orig_np * 255, 0, 255).astype(np.uint8)
        blended = (0.4 * heatmap + 0.6 * orig_np).astype(np.uint8)

        img = Image.fromarray(blended)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print(f"[XRAY] Grad-CAM failed: {e}")
        return None


def predict(image_bytes: bytes) -> dict:
    loaded, device = _get_models()
    if not loaded:
        raise RuntimeError("No X-Ray models loaded")

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = _TRANSFORM(image).unsqueeze(0).to(device)
    tensor_flip = torch.flip(tensor, dims=[3])
    batch = torch.cat([tensor, tensor_flip], dim=0)

    all_probs, per_model = [], {}
    with torch.inference_mode():
        for name, model in loaded.items():
            out = model(batch).cpu().numpy()
            p = out.mean(axis=0)
            per_model[name] = p.tolist()
            all_probs.append(p)

    stacked = np.stack(all_probs, axis=0)
    ensemble = stacked.mean(axis=0)
    model_agreement = float(1.0 - stacked.std(axis=0).mean())

    detected = [(TARGET_COLS[i], float(ensemble[i])) for i in range(NUM_CLASSES) if ensemble[i] >= THRESHOLD]

    if detected:
        prediction, confidence = max(detected, key=lambda x: x[1])
    else:
        top_idx = int(np.argmax(ensemble))
        prediction = TARGET_COLS[top_idx]
        confidence = float(ensemble[top_idx])

    gradcam_b64 = None
    if "DenseNet121" in loaded:
        pred_idx = TARGET_COLS.index(prediction)
        gradcam_b64 = _gradcam_densenet(loaded["DenseNet121"], tensor.clone(), pred_idx)

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "detected_conditions": [{"name": n, "probability": round(p, 4)} for n, p in detected],
        "all_probabilities": {TARGET_COLS[i]: round(float(ensemble[i]), 4) for i in range(NUM_CLASSES)},
        "per_model": {k: {TARGET_COLS[i]: round(v, 4) for i, v in enumerate(probs)} for k, probs in per_model.items()},
        "model_agreement": round(model_agreement, 4),
        "gradcam_image": gradcam_b64,
    }
