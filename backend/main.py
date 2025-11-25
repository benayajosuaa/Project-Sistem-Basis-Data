# === main.py ===
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_core import search_recipes
import asyncio

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskPayload(BaseModel):
    question: str
    top_k: int = 3

@app.post("/ask")
async def ask(payload: AskPayload):
    # Panggil search_recipes (backend + AI)
    result = await asyncio.to_thread(
        search_recipes,
        payload.question,
        payload.top_k
    )

    if not result:
        return {"answer": "Maaf, resep tidak ditemukan.", "results": []}

    first = result[0]

    # Cek error dari rag_core
    if "error" in first:
         return {"answer": f"Terjadi kesalahan: {first['error']}", "results": []}

    # Hasil sukses
    return {
        "answer": first["text"], # Ini teks Markdown dari Gemini
        "results": result
    }

@app.get("/health")
async def health():
    return {"status": "ok"}