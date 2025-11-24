from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "recipes")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

model = SentenceTransformer(EMBED_MODEL)

client = QdrantClient(url=QDRANT_URL)


def embed_text(text: str):
    return model.encode(text).tolist()


def search_recipes(query: str, top_k: int = 3):
    query_vector = embed_text(query)

    result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
    )

    hits = []
    for r in result:
        hits.append({
            "score": r.score,
            "text": r.payload.get("text", ""),
            "recipe_name": r.payload.get("recipe_name", "Untitled")
        })

    return hits
