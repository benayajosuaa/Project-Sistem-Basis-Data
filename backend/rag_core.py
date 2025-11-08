from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import requests

# Load env
load_dotenv()

# Konfigurasi
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "recipes")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# Init
client = QdrantClient(QDRANT_URL)
model = SentenceTransformer(EMBED_MODEL)

def ask_gemini(prompt: str) -> str:
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY
    }
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        r = requests.post(GEMINI_ENDPOINT, headers=headers, json=data)
        r.raise_for_status()
        j = r.json()
        return j["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"[Gemini API Error] {e}"

def query_recipes(user_query: str, top_k: int = 3):
    q_vector = model.encode([user_query])[0]
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=q_vector.tolist(),
        limit=top_k
    )

    if not results:
        return None, None

    context = "\n\n".join(
        f"{r.payload.get('recipe_name', 'Unknown')}\n{r.payload.get('directions', '')}"
        for r in results
    )

    prompt = f"""
Kamu adalah asisten masak yang ringkas dan jelas.
Gunakan konteks resep berikut untuk menjawab.

Konteks resep:
{context}

Pertanyaan pengguna:
\"\"\"{user_query}\"\"\"
"""
    answer = ask_gemini(prompt)
    return answer, results
