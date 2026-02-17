import os
import requests
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

GEMINI_MODEL = "models/gemini-2.5-flash"

GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/"
    f"{GEMINI_MODEL}:generateContent"
)


class ResponseGenerator:
    def __init__(self, temperature: float = 0.3):
        self.temperature = temperature

    def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        if not context_chunks:
            return "I could not find the answer in the provided documents."

        context_text = "\n\n".join(context_chunks)

        prompt = f"""
You are a document intelligence assistant.

Rules:
- Use ONLY the information from the provided context.
- Do NOT add external knowledge.
- Give a COMPLETE and DETAILED explanation.
- Do NOT stop mid-sentence.
- If the answer is not present, say:
  "I could not find the answer in the provided documents."

Context:
{context_text}

Question:
{query}

Answer (write a full explanation in paragraphs):
"""

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": 900,
            },
        }

        response = requests.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            json=payload,
            timeout=40,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Gemini API Error {response.status_code}: {response.text}"
            )

        result = response.json()

        return result["candidates"][0]["content"]["parts"][0]["text"].strip()


# -------------------------
# Local Test
# -------------------------
if __name__ == "__main__":
    chunks = [
        "The termination clause allows either party to end the agreement with a 30-day written notice.",
        "Confidentiality obligations remain valid even after termination.",
    ]

    query = "What does the termination clause say?"

    generator = ResponseGenerator()
    print(generator.generate_answer(query, chunks))
