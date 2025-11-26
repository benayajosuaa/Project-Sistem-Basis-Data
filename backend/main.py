# === main.py (VERIFIED) ===
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio

from rag_core import search_recipes

app = FastAPI(
    title="Recipe Search API",
    version="1.0.0",
    description="RAG-based recipe search"
)

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Saya set wildcard sementara agar development frontend lebih mudah
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskPayload(BaseModel):
    question: str
    top_k: int = 3

@app.get("/")
def root():
    return {"message": "RAG Recipe API is running!"}

@app.post("/ask")
async def ask(payload: AskPayload):
    q = payload.question.strip()
    print(f"🎯 API Request: {q}")

    try:
        # Menjalankan fungsi sinkronus di thread terpisah agar API tidak blocking
        results = await asyncio.to_thread(search_recipes, q, payload.top_k)

        # Cek Error dari rag_core
        if results and isinstance(results[0], dict) and "error" in results[0]:
            return {
                "answer": "Maaf, terjadi kesalahan teknis saat mencari.",
                "results": [],
                "debug_error": results[0]["error"]
            }

        if not results:
            return {"answer": "Tidak ada hasil ditemukan.", "results": []}

        # Ambil jawaban utama
        answer_text = results[0].get("text", "")

        return {
            "answer": answer_text,
            "results": results
        }

    except Exception as e:
        print(f"🚨 ERROR API: {e}")
        return {"answer": f"Error: {e}", "results": []}

# Compatibility endpoint that returns a plain list of hits
# Useful for CLI tooling like `jq -r '.[0].text'`
@app.post("/ask_raw")
async def ask_raw(payload: AskPayload):
    q = payload.question.strip()
    try:
        results = await asyncio.to_thread(search_recipes, q, payload.top_k)
        # Always return a list; if no results, return empty list
        if not isinstance(results, list):
            # Normalize unexpected shapes to a list
            results = [results] if results else []
        return results
    except Exception as e:
        # On error, return a single-item list describing the error to keep shape stable
        return [{
            "recipe_name": "Error",
            "text": f"Error: {e}",
            "score": 0.0,
            "error": str(e)
        }]