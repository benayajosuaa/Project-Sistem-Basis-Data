from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "recipes")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

_model = None
_client = None

def get_model():
    global _model
    if _model is None:
        print("Loading embedding model...")
        _model = SentenceTransformer(EMBED_MODEL)
    return _model

def get_client():
    global _client
    if _client is None:
        print("Connecting to Qdrant...")
        _client = QdrantClient(url=QDRANT_URL)
    return _client

def embed_text(text: str):
    model = get_model()
    return model.encode(text).tolist()

def search_recipes(query: str, top_k: int = 3):
    try:
        print("DEBUG: starting embed")
        query_vector = embed_text(query)
        print("DEBUG: embedding OK", len(query_vector))

        client = get_client()
        print("DEBUG: client OK")

        result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k
        )
        print("DEBUG: search done")

        hits = []
        for r in result:
            hits.append({
                "score": r.score,
                "text": r.payload.get("text", ""),
                "recipe_name": r.payload.get("recipe_name", "Untitled"),
            })

        print("DEBUG: hits OK")
        return hits

    except Exception as e:
        print("ERROR in search:", e)
        return [{"error": str(e)}]

