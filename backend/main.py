from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_core import search_recipes
import asyncio

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
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
    print(f"🎯 API - Received question: '{payload.question}'")
    
    try:
        # Call search_recipes (backend + AI)
        result = await asyncio.to_thread(
            search_recipes,
            payload.question,
            payload.top_k
        )

        if not result:
            print("❌ API - No results found")
            return {"answer": "Maaf, resep tidak ditemukan.", "results": []}

        first = result[0]

        # Check error from rag_core
        if "error" in first:
            print(f"🚨 API - Error in result: {first['error']}")
            return {"answer": f"Terjadi kesalahan: {first['error']}", "results": []}

        print(f"✅ API - Successfully processed, returning {len(result)} results")
        
        # Success response
        return {
            "answer": first["text"],  # This is Markdown text from Gemini/local
            "results": result
        }

    except Exception as e:
        print(f"🚨 API - Unexpected error: {e}")
        return {"answer": f"Terjadi kesalahan sistem: {str(e)}", "results": []}

@app.get("/health")
async def health():
    return {"status": "ok", "service": "recipe-search-api"}

@app.get("/")
async def root():
    return {"message": "Recipe Search API is running!"}