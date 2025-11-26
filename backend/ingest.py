# === ingest.py (VERSI KHUSUS JSON KAMU) ===
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pathlib import Path
import uuid
import os
import time
import sys
import json

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "recipes")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
FILE_NAME = "/Users/benayasimamora/Documents/coding/Project-Sistem-Basis-Data/backend/data/recipe_new.json"
DATA_PATH = Path(FILE_NAME)

print(f"Connecting to: {QDRANT_URL}")
model = SentenceTransformer(EMBED_MODEL)

try:
    client = QdrantClient(url=QDRANT_URL, prefer_grpc=False)
    client.get_collections()
except Exception as e:
    print(f"❌ Gagal connect Qdrant: {e}")
    sys.exit(1)

def recreate_collection():
    try:
        client.delete_collection(COLLECTION_NAME)
        print("🗑️ Collection lama dihapus.")
    except: pass
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    print(f"✨ Collection '{COLLECTION_NAME}' siap!")

def format_nutrients(nutrients_data):
    """Mengubah object nutrients jadi string rapi"""
    if not isinstance(nutrients_data, dict):
        return ""
    # Contoh hasil: "Fat: 18g, Protein: 4g..."
    return ", ".join([f"{k}: {v}" for k, v in nutrients_data.items()])

def ingest(batch_size: int = 256):
    if not DATA_PATH.exists():
        print("❌ File JSON tidak ketemu!")
        return

    print("🔍 Membaca file JSON...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    recreate_collection()
    points = []
    
    print(f"🚀 Memproses {len(raw_data)} resep...")

    for item in raw_data:
        # 1. Ambil Nama
        name = item.get("name") or item.get("Title") or "Tanpa Judul"
        
        # 2. Ambil Steps (Array -> String)
        steps_raw = item.get("steps") or item.get("instructions") or []
        if isinstance(steps_raw, list):
            steps_text = " ".join(steps_raw)
        else:
            steps_text = str(steps_raw)

        # 3. Ambil Nutrisi (Object -> String)
        nutrients_raw = item.get("nutrients") or {}
        nutrients_text = format_nutrients(nutrients_raw)

        # 4. Gabungkan jadi satu teks untuk AI
        # Format ini memudahkan Gemini/RAG nanti membacanya
        full_text = (
            f"{name}\n\n"
            f"Cara Memasak:\n{steps_text}\n\n"
            f"Informasi Nutrisi:\n{nutrients_text}"
        )

        # Buat Vector
        vector = model.encode(full_text).tolist()

        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "recipe_name": name,
                "text": full_text # Payload berisi resep + nutrisi lengkap
            }
        ))

    # Upload
    total = len(points)
    batches = [points[i:i + batch_size] for i in range(0, total, batch_size)]
    
    uploaded = 0
    print(f"📤 Uploading {total} items...")
    for batch in batches:
        try:
            client.upsert(collection_name=COLLECTION_NAME, points=batch)
            uploaded += len(batch)
            print(f"  ✓ {uploaded}/{total}")
        except Exception as e:
            print(f"  ❌ Gagal: {e}")
            time.sleep(1)

    print("\n✅ SELESAI! Data resep + nutrisi berhasil masuk.")

if __name__ == "__main__":
    ingest()