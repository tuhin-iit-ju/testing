from groq import Groq
from config import settings

_client: Groq | None = None

SYSTEM_PROMPT = """You are UyeCare AI, a compassionate and knowledgeable clinical decision support assistant.

Guidelines:
- You assist patients and doctors with health-related questions
- Always recommend consulting a qualified physician for diagnosis and treatment
- Never prescribe medications, dosages, or specific treatments
- Explain medical concepts in simple, clear language
- When a patient shares test results, provide context and general guidance
- Be empathetic, supportive, and non-alarmist
- If the user is a doctor, you may use more technical language
- Always remind users that AI results require physician validation

You serve patients in Bangladesh and are aware of local healthcare context.
"""


def _get_client() -> Groq:
    global _client
    if _client is None:
        if not settings.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY not configured. Add it to your .env file.")
        _client = Groq(api_key=settings.GROQ_API_KEY)
    return _client


def chat(messages: list[dict], context: str | None = None) -> str:
    client = _get_client()

    system = SYSTEM_PROMPT
    if context:
        system += f"\n\nRecent diagnostic context for this session:\n{context}"

    full_messages = [{"role": "system", "content": system}] + messages

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=full_messages,
        max_tokens=1024,
        temperature=0.7,
    )
    return response.choices[0].message.content


def generate_symptom_prescription(prediction: str, top3: list, confidence: float,
                                   filled: int, total: int, risk_level: str) -> str:
    client = _get_client()

    top3_text = "\n".join(
        f"  - {d['disease']}: {d['probability']}%" for d in top3
    )

    prompt = f"""A patient completed an AI symptom assessment on UyeCare. Here are the results:

Primary predicted condition: {prediction} (confidence: {round(confidence * 100, 1)}%)
Top 3 predicted conditions:
{top3_text}
Symptom indicators reported: {filled} of {total}
Risk level: {risk_level}

Write a clear, patient-friendly clinical prescription in the following structure:

1. What your symptoms suggest
   Briefly explain what the top prediction means in plain language.

2. Recommended diagnostic tests
   List the specific tests the patient should get. For each test, write one sentence explaining why it is needed. Use these test names where relevant: ECG, Chest X-Ray, CT Scan, Skin Examination, Blood Test, Blood Pressure Monitoring.

3. How soon should you see a doctor?
   Choose one: Today (urgent) / Within 3–7 days / At your next routine appointment. Explain why.

4. What to watch for
   List 2–3 warning symptoms that should prompt the patient to seek immediate care.

5. Immediate steps
   One or two simple actions the patient can take right now (rest, monitor vitals, avoid exertion, etc.).

Write in warm, clear language. No medication names or dosages. Keep the total response under 320 words."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=600,
        temperature=0.5,
    )
    return response.choices[0].message.content


def generate_recommendation(test_type: str, prediction: str, confidence: float, risk_level: str) -> str:
    client = _get_client()

    prompt = f"""A patient just received the following AI diagnostic result:
- Test type: {test_type}
- Prediction: {prediction}
- Confidence: {confidence * 100:.1f}%
- Risk level: {risk_level}

Please provide:
1. A brief, plain-language explanation of what this result means
2. General lifestyle or monitoring advice (no drug names)
3. Urgency of seeing a doctor (routine / within a week / urgent)

Keep the response under 200 words. Be empathetic and clear."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=512,
        temperature=0.5,
    )
    return response.choices[0].message.content
