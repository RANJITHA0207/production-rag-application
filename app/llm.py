import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


class LLMClient:

    def __init__(self):

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Please check your .env file."
            )

        self.client = OpenAI(
            api_key=api_key
        )

    def generate_answer(self, question, context):

        prompt = f"""
You are a helpful question-answering assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer is not available in the context, say:
"I don't have enough information in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

        try:

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You answer questions using "
                            "provided document context."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0
            )

            return response.choices[0].message.content

        except Exception as error:

            return (
                "LLM generation is currently unavailable. "
                f"Reason: {error}"
            )