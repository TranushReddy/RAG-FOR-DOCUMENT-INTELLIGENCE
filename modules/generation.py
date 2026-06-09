import os
import requests
from typing import List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --------------------------------------------------
# Configuration
# --------------------------------------------------

AI_API_PROVIDER = os.getenv("AI_API_PROVIDER", "groq").strip().lower()

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")

GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/"
    f"{GEMINI_MODEL}:generateContent"
)

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile"
)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


# --------------------------------------------------
# Response Generator
# --------------------------------------------------

class ResponseGenerator:
    def __init__(
        self,
        temperature: float = 0.3,
        provider: str | None = None
    ):
        self.temperature = temperature
        self.provider = (provider or AI_API_PROVIDER).strip().lower()

        if self.provider == "gemini":
            if not GEMINI_API_KEY:
                raise ValueError(
                    "GEMINI_API_KEY not found in .env"
                )

        elif self.provider == "groq":
            if not GROQ_API_KEY:
                raise ValueError(
                    "GROQ_API_KEY not found in .env"
                )

        else:
            raise ValueError(
                "Unsupported provider. Use 'gemini' or 'groq'."
            )

    def generate_answer(
        self,
        query: str,
        context_chunks: List[str]
    ) -> str:

        if not context_chunks:
            return (
                "I could not find the answer "
                "in the provided documents."
            )

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

Answer:
"""

        if self.provider == "gemini":
            return self._generate_with_gemini(prompt)

        return self._generate_with_groq(prompt)

    # --------------------------------------------------
    # Gemini
    # --------------------------------------------------

    def _generate_with_gemini(self, prompt: str) -> str:

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
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
                f"Gemini API Error "
                f"{response.status_code}: "
                f"{response.text}"
            )

        result = response.json()

        try:
            return (
                result["candidates"][0]
                ["content"]["parts"][0]["text"]
                .strip()
            )

        except Exception:
            raise RuntimeError(
                f"Unexpected Gemini response:\n{result}"
            )

    # --------------------------------------------------
    # Groq
    # --------------------------------------------------

    def _generate_with_groq(self, prompt: str) -> str:

        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": self.temperature,
            "max_tokens": 900
        }

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            GROQ_URL,
            json=payload,
            headers=headers,
            timeout=40,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"GROQ API Error "
                f"{response.status_code}: "
                f"{response.text}"
            )

        result = response.json()

        try:
            return (
                result["choices"][0]
                ["message"]["content"]
                .strip()
            )

        except Exception:
            raise RuntimeError(
                f"Unexpected GROQ response:\n{result}"
            )


# --------------------------------------------------
# Local Test
# --------------------------------------------------

if __name__ == "__main__":

    chunks = [
        "The termination clause allows either party "
        "to end the agreement with a 30-day written notice.",

        "Confidentiality obligations remain valid "
        "even after termination."
    ]

    query = "What does the termination clause say?"

    generator = ResponseGenerator()

    answer = generator.generate_answer(
        query,
        chunks
    )

    print("\nGenerated Answer:\n")
    print(answer)
