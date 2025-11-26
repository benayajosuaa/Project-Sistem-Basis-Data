# === rag_core.py (FINAL & VERIFIED) ===
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import re
import time

load_dotenv()

# --- CONFIG ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "recipes")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

# --- SETUP AI (Gemini) ---
genai_available = False
try:
    import google.generativeai as genai
    
    # Validasi versi library secara runtime
    if not hasattr(genai, 'configure'):
        print("⚠️  Versi google-generativeai usang. AI dimatikan.")
        print("    -> Solusi: pip install -U google-generativeai")
    elif GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        genai_available = True
        print("👉 Google Gemini API configured.")
    else:
        print("⚠️  GOOGLE_API_KEY tidak ditemukan. Fitur AI Generative dimatikan.")

except ImportError:
    print("⚠️  Library 'google.generativeai' belum diinstall.")
except Exception as e:
    print(f"⚠️  Error setup Gemini: {e}")


# --- CACHE ---
_model = None
_client = None

def get_model():
    global _model
    if _model is None:
        print(f"Loading embedding model: {EMBED_MODEL} ...")
        _model = SentenceTransformer(EMBED_MODEL)
    return _model

def get_client():
    global _client
    if _client is None:
        # print(f"👉 Connecting to Qdrant: {QDRANT_URL}")
        # prefer_grpc=False wajib
        _client = QdrantClient(url=QDRANT_URL, prefer_grpc=False)
    return _client

def embed_text(text: str):
    return get_model().encode(text).tolist()


# ==========================================
#  LOCAL FORMATTER (Fallback)
# ==========================================
_MEASURE_RE = re.compile(r'\b(cup|cups|tbsp|tsp|gram|g|kg|ml|l|oz|siung|buah|lembar|ikat|batang|sendok)\b', re.I)
_INSTR_KEYWORDS = [
    "panaskan", "campur", "aduk", "masak", "goreng", "rebus", "tumis",
    "bakar", "potong", "iris", "haluskan", "sajikan", "mix", "cook", "bake"
]

def local_format_to_markdown(raw_text: str, recipe_name: str) -> str:
    print(f"🔧 LOCAL FORMATTER: {recipe_name}")
    text = raw_text.strip()
    
    if any(k in text for k in ["Bahan-bahan", "Cara Memasak", "Ingredients", "Instructions"]):
        if not text.startswith("#"):
            return f"## {recipe_name}\n\n{text}"
        return text
    
    return parse_unstructured_text(text, recipe_name)

def parse_unstructured_text(text: str, recipe_name: str) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    ingredients = []
    steps = []
    
    for s in sentences:
        s_strip = s.strip()
        if not s_strip: continue
        
        if (_MEASURE_RE.search(s_strip) or re.match(r'^\d+(\.\d*)?\s*', s_strip)):
            ingredients.append(s_strip)
        elif any(k in s_strip.lower() for k in _INSTR_KEYWORDS):
            steps.append(s_strip)
        else:
            steps.append(s_strip)
    
    md = [f"## {recipe_name}\n"]
    md.append("### 🛒 Bahan-bahan")
    if ingredients:
        for ing in ingredients:
            md.append(f"- {ing}")
    else:
        md.append("- (Lihat detail pada deskripsi)")

    md.append("\n### 🍳 Cara Memasak")
    if steps:
        for i, step in enumerate(steps, 1):
            md.append(f"{i}. {step}")
    else:
        md.append(f"1. {text}") 
        
    return "\n".join(md)


# ==========================================
#  AI FORMATTER (Gemini)
# ==========================================
def format_with_gemini(raw_text: str, recipe_name: str) -> str:
    if not genai_available:
        return local_format_to_markdown(raw_text, recipe_name)

    print(f"🔍 GEMINI Processing: {recipe_name}")
    
    cleaned_text = raw_text.strip()
    if cleaned_text.lower().startswith(recipe_name.lower()):
        cleaned_text = cleaned_text[len(recipe_name):].lstrip(".:- ").strip()

    prompt = f"""
Perbaiki format resep ini menjadi Markdown bahasa Indonesia.
Judul: {recipe_name}

Format output wajib:
## {recipe_name}

### 🛒 Bahan-bahan
- (daftar bahan)

### 🍳 Cara Memasak
1. (langkah-langkah)

### ℹ️ Nutrisi
- (jika ada, kalau tidak tulis 'Tidak tersedia')

Teks:
{cleaned_text}
"""

    tried_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
    
    for model_name in tried_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt,
                generation_config={"temperature": 0.2}
            )
            result_text = response.text.strip()
            
            if "###" in result_text:
                print(f"✅ GEMINI Success ({model_name})")
                return result_text
        except Exception:
            continue

    print("🔧 GEMINI Gagal. Fallback ke Local.")
    return local_format_to_markdown(raw_text, recipe_name)


# ==========================================
#  SEARCH FUNCTION
# ==========================================
def search_recipes(query: str, top_k: int = 3):
    try:
        print(f"\n🔎 SEARCH: '{query}'")
        
        vector = embed_text(query)
        client = get_client()

        search_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=top_k,
            with_payload=True
        )

        hits = []
        points = search_result.points

        if not points:
            return []

        for i, point in enumerate(points):
            raw_text = point.payload.get("text", "")
            recipe_name = point.payload.get("recipe_name", "Tanpa Judul")
            score = point.score
            
            print(f"   [{i+1}] {recipe_name} ({score:.3f})")

            # AI Format hanya untuk Top 1 (hemat waktu)
            if i == 0:
                formatted_text = format_with_gemini(raw_text, recipe_name)
            else:
                formatted_text = local_format_to_markdown(raw_text, recipe_name)

            hits.append({
                "recipe_name": recipe_name,
                "text": formatted_text,
                "score": score,
            })

        return hits

    except Exception as e:
        print(f"🚨 ERROR SEARCH: {e}")
        return [{"error": str(e)}]