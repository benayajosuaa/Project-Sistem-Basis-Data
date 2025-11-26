# === ingest.py (VERSI KHUSUS JSON KAMU) ===
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from sentence_transformers import SentenceTransformer
# import rag_core helper if available
try:
    from backend import rag_core
except Exception:
    rag_core = None
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
    """Mengubah object nutrients jadi list items"""
    if not isinstance(nutrients_data, dict):
        return []
    # Return as list of individual items for cleaner markdown formatting
    return [f"{k}: {v}" for k, v in nutrients_data.items()]

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
        
        # 2. Ambil Steps (simpan bentuk asli sebagai list langkah-langkah)
        steps_raw = item.get("steps") or item.get("instructions") or []
        if isinstance(steps_raw, list):
            # Each step is a separate string in the list
            steps_list = [str(s).strip() for s in steps_raw if str(s).strip()]
        else:
            # If it's a single string, try to split by "; " or newlines
            steps_list = [s.strip() for s in str(steps_raw).split(";") if s.strip()]
        
        # Join steps with semicolon for text embedding
        steps_text = "; ".join(steps_list)

        # 3. Ambil Nutrisi (Object -> List)
        nutrients_raw = item.get("nutrients") or {}
        nutrients_list = format_nutrients(nutrients_raw)

        # 4. Buat teks untuk embedding (tanpa mencampur section)
        # Format: Nama | Steps | Nutrition (ingredients akan diekstrak dari steps)
        parts = [name, ""]
        parts.append("Langkah-langkah:")
        parts.append(steps_text)
        parts.append("")
        if nutrients_list:
            parts.append("Nutrisi:")
            parts.append(", ".join(nutrients_list))

        full_text = "\n".join(parts)

        # Buat Vector
        vector = model.encode(full_text).tolist()

        payload = {
            "recipe_name": name,
            "text": full_text,
            "steps": steps_list,  # Store as list for easy processing
            "steps_text": steps_text,  # Store joined version for backward compatibility
            "nutrients": nutrients_list,  # Store as list of items
        }
        
        # Try to extract ingredients from steps if not available
        if not payload.get("ingredients"):
            # If rag_core available and Gemini online, try to extract structured ingredients
            try:
                if rag_core is not None:
                    extracted = rag_core.llm_extract_structured(full_text, name, target_language='indonesian')
                    ings = extracted.get('ingredients') or []
                    instrs = extracted.get('instructions') or []
                    text_id = extracted.get('text_id') or ''
                    if ings:
                        payload['ingredients'] = [str(x).strip() for x in ings if str(x).strip()]
                    if instrs:
                        payload['instructions'] = [str(x).strip() for x in instrs if str(x).strip()]
                    if text_id:
                        payload['text_id'] = text_id
            except Exception as e:
                # silently ignore LLM failures and continue with no ingredients
                print(f"⚠️ LLM extraction failed for '{name}': {e}")

        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload=payload
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