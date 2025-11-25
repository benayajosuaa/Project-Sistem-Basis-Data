from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pathlib import Path
import uuid
import os

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "recipes")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

model = SentenceTransformer(EMBED_MODEL)

client = QdrantClient(url=QDRANT_URL)

DATA_PATH = Path("/app/data/final_recipes_for_rag.txt")


def create_collection():
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print("Collection created!")


def ingest():
    if not DATA_PATH.exists():
        raise FileNotFoundError("File recipes_final.txt tidak ditemukan")

    create_collection()

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    points = []
    for line in lines:
            line = line.strip()
            if not line:
                continue

            if ":" not in line:
                print("SKIPPING BROKEN LINE:", line)
                continue

            recipe_name, text = line.split(":", 1)

            recipe_name = recipe_name.strip()
            text = text.strip()

            if not text:
                print("SKIP EMPTY TEXT:", line)
                continue

            vector = model.encode(text).tolist()

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "recipe_name": recipe_name,
                        "text": text
                    }
                )
            )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Ingest selesai: {len(points)} data dimasukkan.")


if __name__ == "__main__":
    ingest()
