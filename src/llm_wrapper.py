import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("Please set GROQ_API_KEY in your .env file")

client = Groq(api_key=GROQ_API_KEY)

class GroqLLM:
    """Simple wrapper for Groq chat model"""
    def __init__(self, model="llama-3.1-8b-instant"):
        self.model = model

    def __call__(self, prompt: str) -> str:
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
