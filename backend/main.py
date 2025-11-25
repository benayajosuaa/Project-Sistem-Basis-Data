from fastapi import FastAPI
from pydantic import BaseModel
from rag_core import search_recipes
import asyncio

app = FastAPI()

class AskPayload(BaseModel):
    question: str
    top_k: int = 3

@app.post("/ask")
async def ask(payload: AskPayload):
    try:
        # Jalankan search di thread pool
        result = await asyncio.to_thread(
            search_recipes,
            payload.question,
            payload.top_k
        )

        # Kalau function search balikin None atau list kosong
        if not result:
            return {
                "answer": "Tidak ada jawaban ditemukan.",
                "results": []
            }

        first = result[0]

        # Kalau result berisi error dict seperti {"error": "..."}
        if isinstance(first, dict) and "error" in first:
            return {
                "answer": "Terjadi error saat mencari jawaban.",
                "results": result
            }

        # Normal case: item punya payload text
        if isinstance(first, dict) and "text" in first:
            answer = first["text"]
        else:
            # fallback kalau format item aneh
            answer = str(first)

        return {
            "answer": answer,
            "results": result
        }

    except Exception as e:
        # error handling global
        return {"error": str(e)}

@app.get("/health")
async def health():
    return {"status": "ok"}
