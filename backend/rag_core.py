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
        print("Loading embedding model... (this may download model files if not cached)")
        try:
            _model = SentenceTransformer(EMBED_MODEL)
        except Exception as e:
            print("ERROR loading embedding model:", e)
            raise
    return _model

def get_client():
    global _client
    if _client is None:
        print("Connecting to Qdrant at", QDRANT_URL)
        # Try a few times in case Qdrant is still starting or DNS transient error
        import time
        last_exc = None
        for attempt in range(1, 6):
            try:
                _client = QdrantClient(url=QDRANT_URL)
                # quick check to ensure connection works
                _client.get_collections()
                print("Connected to Qdrant (attempt", attempt, ")")
                last_exc = None
                break
            except Exception as e:
                print(f"Attempt {attempt} failed to connect to Qdrant:", e)
                last_exc = e
                time.sleep(attempt)  # backoff
        if last_exc is not None:
            print("Failed to connect to Qdrant after retries:", last_exc)
            raise last_exc
    return _client

def embed_text(text: str):
    model = get_model()
    return model.encode(text).tolist()

def search_recipes(query: str, top_k: int = 3):
    try:
        print("DEBUG: starting search with query:", query)
        query_vector = embed_text(query)
        print("DEBUG: embedding OK", len(query_vector))
        
        client = get_client()
        print("DEBUG: client OK")
        
        # Gunakan query_points dengan vector yang sudah di-embed
        result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k
        )
        print("DEBUG: search done")
        
        hits = []
        # result adalah QueryResponse dengan field 'points'
        for point in result.points:
            hits.append({
                "score": point.score,
                "text": point.payload.get("text", ""),
                "recipe_name": point.payload.get("recipe_name", "Untitled"),
            })
        print("DEBUG: hits OK")
        return hits
        
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("ERROR in search:", e)
        print(tb)
        # Return structured error with type and message (so frontend can show it)
        err_msg = f"{type(e).__name__}: {str(e)}"
        return [{"error": err_msg}]