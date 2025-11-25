# === rag_core.py ===
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

# --- PERUBAHAN DI SINI (KITA PAKSA LOCALHOST) ---
# Jangan pakai os.getenv dulu biar pasti jalan
QDRANT_URL = "http://localhost:6333" 
# -----------------------------------------------

COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "recipes")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Setup Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model_gemini = genai.GenerativeModel('gemini-1.5-flash')
else:
    model_gemini = None

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
        # Debugging Print: Biar kita lihat dia connect kemana
        print(f"👉 MENCOBA CONNECT KE: {QDRANT_URL}") 
        
        try:
            _client = QdrantClient(url=QDRANT_URL)
            # Cek koneksi beneran nyambung atau enggak
            _client.get_collections() 
            print("✅ BERHASIL CONNECT KE QDRANT!")
        except Exception as e:
            print(f"❌ GAGAL CONNECT: {e}")
            raise e
            
    return _client

def embed_text(text: str):
    return get_model().encode(text).tolist()

def format_with_gemini(raw_text: str, recipe_name: str):
    if not model_gemini:
        return raw_text

    prompt = f"""
    Kamu adalah asisten koki. Format ulang resep ini ke MARKDOWN yang rapi.
    
    Judul: {recipe_name}
    Konten: "{raw_text}"

    Aturan:
    1. Buat bagian "### 🛒 Bahan-bahan" (bullet points).
    2. Buat bagian "### 🍳 Cara Memasak" (nomor 1, 2, 3...).
    3. JANGAN mengubah angka takaran.
    4. Langsung ke konten.
    """
    try:
        response = model_gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini Error: {e}")
        return raw_text

def search_recipes(query: str, top_k: int = 3):
    try:
        print("DEBUG: searching:", query)
        query_vec = embed_text(query)

        client = get_client()
        result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            limit=top_k,
        )

        hits = []
        for i, point in enumerate(result.points):
            raw = point.payload.get("text", "").strip()
            recipe_name = point.payload.get("recipe_name", "Resep")

            if i == 0:
                final_text = format_with_gemini(raw, recipe_name)
            else:
                final_text = raw 

            hits.append({
                "score": point.score,
                "text": final_text,
                "recipe_name": recipe_name,
            })

        return hits

    except Exception as e:
        import traceback
        print("ERROR:", e)
        return [{"error": f"{type(e).__name__}: {e}"}]