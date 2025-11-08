from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_core import query_recipes
import uvicorn

# Inisialisasi FastAPI
app = FastAPI(title="RAG Resep API", version="1.0.0")

# ===== Tambahkan CORS agar bisa diakses frontend Next.js =====
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti ke ["http://localhost:3000"] jika hanya untuk lokal dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model input untuk request POST /ask
class AskRequest(BaseModel):
    question: str
    top_k: int = 3

# Endpoint root (cek apakah API hidup)
@app.get("/")
def root():
    return {"message": "RAG Resep API aktif!"}

# Endpoint utama untuk RAG query
@app.post("/ask")
def ask(req: AskRequest):
    try:
        # Panggil fungsi pencarian resep di rag_core.py
        answer, results = query_recipes(req.question, req.top_k)

        # Kalau tidak ada hasil
        if not results:
            return {
                "message": "Tidak ada hasil mirip ditemukan di database.",
                "answer": answer or "Tidak ada jawaban dari model."
            }

        # Format hasil pencarian
        hits = [
            {
                "score": float(r.score),
                "recipe_name": r.payload.get("recipe_name", "Tanpa nama"),
                "text": (r.payload.get("directions") or "")[:200] + "..."
            }
            for r in results
        ]

        # Return hasil akhir
        return {"answer": answer, "results": hits}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan: {e}")

# Jalankan server FastAPI
if __name__ == "__main__":
    print("🚀 Menjalankan RAG Resep API di http://127.0.0.1:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
