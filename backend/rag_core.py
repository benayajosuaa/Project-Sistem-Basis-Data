# rag_core.py
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

load_dotenv()

# optional LLM (google generative ai)
try:
    import google.generativeai as genai
except Exception:
    genai = None

# --- CONFIG ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "recipes")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# accept either GOOGLE_API_KEY or GEMINI_API_KEY env var
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

# Setup Gemini (if available)
model_gemini = None
if genai is not None and GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        # try the most stable option; library/runtime may change available model names
        # we'll try a small list later when calling generate_content if needed
        model_gemini = genai.GenerativeModel("gemini-pro")
        print("👉 Gemini client configured (requested model: gemini-pro)")
    except Exception as e:
        print("⚠️ Gemini setup failed:", e)
        model_gemini = None
else:
    if genai is None:
        print("⚠️ google.generativeai not installed or import failed.")
    else:
        print("⚠️ GOOGLE_API_KEY not set; Gemini disabled.")


# cache
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
        print(f"👉 MENCOBA CONNECT KE: {QDRANT_URL}")
        _client = QdrantClient(url=QDRANT_URL)
        # check existence lazily (may raise if DB down)
        try:
            _client.get_collections()
            print("✅ BERHASIL CONNECT KE QDRANT!")
        except Exception as e:
            print("⚠️ Warning: get_collections() error:", e)
    return _client

def embed_text(text: str):
    return get_model().encode(text).tolist()


# --- Local heuristic formatter (fallback when LLM fails) ---
import re

_MEASURE_RE = re.compile(r'\b(cup|cups|tbsp|tsp|tablespoon|tablespoons|teaspoon|gram|g|kg|ml|l|ounce|oz|pound|slices?)\b', re.I)
_INSTR_KEYWORDS = [
    "preheat", "mix", "combine", "stir", "bake", "cook",
    "add", "fold", "pour", "simmer", "whisk", "beat", "knead", "heat",
    "remove", "reduce", "return", "cover", "serve"
]

def local_format_to_markdown(raw_text: str, recipe_name: str) -> str:
    text = raw_text.strip()
    # split into sentences
    sentences = re.split(r'(?<=[\.\?\!])\s+', text)
    # try to collect ingredients: sentences that contain measure tokens or numbers
    ingredients = []
    steps = []
    remainder = []

    for s in sentences:
        s_strip = s.strip()
        low = s_strip.lower()
        if not s_strip:
            continue
        # if contains measurement tokens OR starts with a digit -> ingredient candidate
        if _MEASURE_RE.search(s_strip) or re.match(r'^\d+(\.| )', s_strip):
            ingredients.append(s_strip.rstrip('.'))
        # if looks like instruction by keyword -> step
        elif any(k in low for k in _INSTR_KEYWORDS) or len(s_strip.split()) > 10:
            steps.append(s_strip.rstrip('.'))
        else:
            remainder.append(s_strip.rstrip('.'))

    # fallback: if no ingredients found, try to parse from comma separated first sentence
    if not ingredients and remainder:
        first = remainder[0]
        parts = [p.strip() for p in re.split(r',|\band\b', first) if p.strip()]
        # keep only short items
        cand = [p for p in parts if 1 < len(p.split()) <= 6]
        if cand:
            ingredients.extend(cand)
            remainder = remainder[1:]

    # build markdown
    md = [f"## {recipe_name}\n"]
    md.append("### 🛒 Bahan-bahan")
    if ingredients:
        for ing in ingredients:
            md.append(f"- {ing}")
    else:
        md.append("- (Tidak ada daftar bahan terdeteksi dari teks)")

    md.append("\n### 🍳 Cara Memasak")
    if steps:
        for i, st in enumerate(steps, 1):
            md.append(f"{i}. {st}")
    else:
        # fallback: chunk remainder into steps by reasonable lengths
        if remainder:
            for i, st in enumerate(remainder, 1):
                md.append(f"{i}. {st}")
        else:
            md.append("1. (Instruksi tidak tersedia)")

    # optional info
    md.append("\n### ℹ️ Informasi Tambahan")
    # try extract nutrition-like pattern
    nut = re.findall(r'(Total Fat|Saturated Fat|Cholesterol|Sodium|Total Carbohydrate|Dietary Fiber|Total Sugars|Protein|Vitamin|Calcium|Iron|Potassium)[^,;\n]*', raw_text, re.I)
    if nut:
        for n in nut:
            md.append(f"- {n.strip()}")
    else:
        md.append("- (Tidak ada informasi tambahan terdeteksi)")

    return "\n".join(md)


# --- Format with Gemini (if available) ---
def format_with_gemini(raw_text: str, recipe_name: str) -> str:
    if not model_gemini:
        print("ℹ️ Gemini not configured — using local formatter.")
        return local_format_to_markdown(raw_text, recipe_name)

    # prepare strict prompt
    prompt = f"""
Anda adalah asisten koki profesional.
Format ulang teks resep berikut menjadi **Markdown rapi** persis sesuai TEMPLATE:

## {recipe_name}

### 🛒 Bahan-bahan
- (satu bahan per bullet)

### 🍳 Cara Memasak
1. (langkah pertama)
2. (langkah kedua)
...

### ℹ️ Informasi Tambahan
- (kalori/nutrisi jika ada)

ATURAN KERAS:
- Output HARUS mengikuti template di atas.
- Jangan membuat paragraf panjang.
- Jangan menambah penjelasan apapun.
- Jangan merangkum isi; hanya ubah struktur menjadi Markdown.

=== TEKS ASLI ===
{raw_text}

=== OUTPUT ===
"""

    # try multiple model names/order if runtime complains
    tried_models = []
    for candidate_model in ["gemini-pro", "gemini-1.5", "gemini-1.5-flash", "gemini-1.0-pro"]:
        try:
            tried_models.append(candidate_model)
            # re-init model object with candidate name (some SDKs require new object)
            gen_model = genai.GenerativeModel(candidate_model)
            resp = gen_model.generate_content(
                prompt,
                generation_config={"temperature": 0, "max_output_tokens": 2048}
            )
            text = (resp.text or "").strip()
            if text:
                # quick sanity check: did we get markers?
                if ("###" in text) or ("- " in text) or ("1." in text):
                    print(f"✅ Gemini formatted using model {candidate_model}")
                    return text
                else:
                    # if response has no list markers, continue trying others
                    print(f"⚠️ Gemini response from {candidate_model} lacks list markers; trying next model.")
            else:
                print(f"⚠️ Empty response from Gemini model {candidate_model}")
        except Exception as e:
            print(f"⚠️ Gemini candidate {candidate_model} failed: {e}")

    # if we reach here, all LLM attempts failed -> use local fallback
    print("❌ All Gemini attempts failed (tried models: " + ", ".join(tried_models) + "). Using local formatter.")
    return local_format_to_markdown(raw_text, recipe_name)


# --- SEARCH + OUTPUT ---
def search_recipes(query: str, top_k: int = 3):
    try:
        print("DEBUG: searching:", query)
        vec = embed_text(query)
        client = get_client()

        result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vec,
            limit=top_k,
        )

        hits = []
        for i, point in enumerate(result.points):
            raw = point.payload.get("text", "").strip()
            recipe_name = point.payload.get("recipe_name", "Resep")

            # only format top-1 with LLM/fallback
            if i == 0:
                formatted = format_with_gemini(raw, recipe_name)
            else:
                formatted = raw

            hits.append({
                "score": point.score,
                "text": formatted,
                "recipe_name": recipe_name,
            })
        return hits

    except Exception as e:
        print("ERROR in search_recipes:", e)
        return [{"error": f"{type(e).__name__}: {e}"}]
