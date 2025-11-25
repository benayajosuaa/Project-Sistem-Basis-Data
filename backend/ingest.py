# === ingest.py ===
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pathlib import Path
import uuid
import os

load_dotenv()

# Pastikan URL ini localhost (karena script jalan di Mac, DB di Docker)
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "recipes")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

print(f"Connecting to: {QDRANT_URL}")

model = SentenceTransformer(EMBED_MODEL)
client = QdrantClient(url=QDRANT_URL)

# --- PERBAIKAN DISINI ---
# Kita anggap file txt ada di folder yang sama dengan ingest.py
# Ganti nama file sesuai nama file asli Anda!
FILE_NAME = "/Users/benayasimamora/Documents/coding/Project-Sistem-Basis-Data/backend/data/recipes_final.txt" 
DATA_PATH = Path(FILE_NAME) 

def create_collection():
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print(f"Collection '{COLLECTION_NAME}' created/recreated!")

def ingest():
    # Cek apakah file ada
    if not DATA_PATH.exists():
        print(f"❌ ERROR: File '{DATA_PATH.absolute()}' tidak ditemukan!")
        print("Pastikan file .txt ada di folder yang sama dengan ingest.py")
        return

    create_collection()

    print("Membaca file...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    points = []
    print(f"Memproses {len(lines)} baris...")
    
    for line in lines:
            line = line.strip()
            if not line:
                continue

            # Validasi format "Resep : Cara masak"
            if ":" not in line:
                # print("SKIPPING (Format salah):", line)
                continue

            recipe_name, text = line.split(":", 1)
            recipe_name = recipe_name.strip()
            text = text.strip()

            if not text:
                continue

            # Buat Vector (Embedding)
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

    # Upload ke Qdrant
    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"✅ SUKSES! {len(points)} resep berhasil dimasukkan ke database.")
    else:
        print("⚠️ TIDAK ADA DATA yang dimasukkan. Cek format file txt Anda.")

if __name__ == "__main__":
    ingest()