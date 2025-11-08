import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import uuid

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = os.getenv("QDRANT_COLLECTION", "recipes")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

DATA_PATH = Path("data/recipes_final.txt")

def main():
    assert DATA_PATH.exists(), f"File tidak ditemukan: {DATA_PATH}"

    # Load model embedding
    model = SentenceTransformer(EMBED_MODEL)
    dim = model.get_sentence_embedding_dimension()

    # Koneksi ke Qdrant
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # Buat koleksi kalau belum ada
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION not in collections:
        client.recreate_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    # Baca data resep
    with DATA_PATH.open("r", encoding="utf-8") as f:
        docs = [line.strip() for line in f if line.strip()]

    # Generate embedding
    embeddings = model.encode(docs, convert_to_numpy=True, show_progress_bar=True)

    # Upload ke Qdrant
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec.astype(np.float32).tolist(),
            payload={"text": text},
        )
        for text, vec in zip(docs, embeddings)
    ]

    client.upsert(collection_name=COLLECTION, points=points)
    print(f"Sukses ingest {len(points)} resep ke koleksi '{COLLECTION}'.")

if __name__ == "__main__":
    main()
