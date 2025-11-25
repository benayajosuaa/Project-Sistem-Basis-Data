from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import re

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
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

# Setup Gemini (if available)
model_gemini = None
if genai is not None and GOOGLE_API_KEY:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
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
        try:
            _client.get_collections()
            print("✅ BERHASIL CONNECT KE QDRANT!")
        except Exception as e:
            print("⚠️ Warning: get_collections() error:", e)
    return _client

def embed_text(text: str):
    return get_model().encode(text).tolist()


# --- Local heuristic formatter (IMPROVED) ---
_MEASURE_RE = re.compile(r'\b(cup|cups|tbsp|tsp|tablespoon|tablespoons|teaspoon|gram|g|kg|ml|l|ounce|oz|pound|slices?)\b', re.I)
_INSTR_KEYWORDS = [
    "preheat", "mix", "combine", "stir", "bake", "cook",
    "add", "fold", "pour", "simmer", "whisk", "beat", "knead", "heat",
    "remove", "reduce", "return", "cover", "serve"
]

def local_format_to_markdown(raw_text: str, recipe_name: str) -> str:
    print(f"🔧 LOCAL FORMATTER - Processing: {recipe_name}")
    
    text = raw_text.strip()
    
    # Special handling for structured text like Apple Pie Filling
    if any(keyword in text for keyword in ["Bahan-bahan", "Cara Memasak", "Informasi Nutrisi"]):
        print("🔧 LOCAL FORMATTER - Detected structured text, using section-based parsing")
        return parse_structured_text(text, recipe_name)
    
    # Fallback to sentence-based parsing for unstructured text
    return parse_unstructured_text(text, recipe_name)

def parse_structured_text(text: str, recipe_name: str) -> str:
    """Parse text that already has sections like Bahan-bahan, Cara Memasak, etc."""
    lines = text.split('\n')
    ingredients = []
    steps = []
    nutrition = []
    
    current_section = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Detect section headers
        if "Bahan-bahan" in line:
            current_section = "ingredients"
            continue
        elif "Cara Memasak" in line:
            current_section = "steps"
            continue
        elif "Informasi Nutrisi" in line:
            current_section = "nutrition"
            continue
            
        # Skip metadata lines in ingredients section
        if current_section == "ingredients" and "Cool for" in line:
            continue
            
        # Process content based on current section
        if current_section == "ingredients" and line:
            ingredients.append(line)
        elif current_section == "steps" and line:
            # Split long text into individual steps
            sentences = re.split(r'(?<=[.!?])\s+', line)
            for sentence in sentences:
                if sentence.strip() and len(sentence.strip()) > 10:  # Only include substantial sentences
                    steps.append(sentence.strip())
        elif current_section == "nutrition" and line:
            nutrition.append(line)
    
    # Build markdown
    md = [f"## {recipe_name}\n"]
    
    # Ingredients section
    md.append("### 🛒 Bahan-bahan")
    if ingredients:
        for ing in ingredients:
            if ing.strip() and ing not in ["Bahan-bahan", "Cara Memasak", "Informasi Nutrisi"]:
                md.append(f"- {ing.strip()}")
    else:
        md.append("- (Bahan tidak terdeteksi)")
    
    # Steps section  
    md.append("\n### 🍳 Cara Memasak")
    if steps:
        for i, step in enumerate(steps, 1):
            md.append(f"{i}. {step}")
    else:
        md.append("1. (Langkah memasak tidak terdeteksi)")
    
    # Nutrition section
    md.append("\n### ℹ️ Informasi Nutrisi")
    if nutrition:
        for nut in nutrition:
            if nut.strip():
                md.append(f"- {nut.strip()}")
    else:
        # Try to extract nutrition info from text
        nutrition_info = extract_nutrition_info(text)
        if nutrition_info:
            for info in nutrition_info:
                md.append(f"- {info}")
        else:
            md.append("- (Informasi nutrisi tidak tersedia)")
    
    result = "\n".join(md)
    print(f"🔧 LOCAL FORMATTER - Structured result:\n{result}")
    return result

def parse_unstructured_text(text: str, recipe_name: str) -> str:
    """Parse unstructured text using sentence analysis."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    ingredients, steps, remainder = [], [], []
    
    for s in sentences:
        s_strip = s.strip()
        if not s_strip: 
            continue
            
        # Check if it's likely an ingredient
        if (_MEASURE_RE.search(s_strip) or 
            re.match(r'^\d+(\.\d*)?\s*', s_strip) or
            any(word in s_strip.lower() for word in ['cup', 'tbsp', 'tsp', 'gram', 'kg', 'ml'])):
            ingredients.append(s_strip.rstrip('.'))
        # Check if it's likely an instruction
        elif any(keyword in s_strip.lower() for keyword in _INSTR_KEYWORDS):
            steps.append(s_strip.rstrip('.'))
        else:
            remainder.append(s_strip.rstrip('.'))
    
    # If no ingredients detected but we have remainder, use first few lines as ingredients
    if not ingredients and remainder:
        # Take first 2-4 items that look like ingredients
        potential_ingredients = []
        for item in remainder[:6]:
            if len(item.split()) <= 8:  # Reasonable length for ingredient
                potential_ingredients.append(item)
        if potential_ingredients:
            ingredients.extend(potential_ingredients)
            remainder = remainder[len(potential_ingredients):]
    
    # Build markdown
    md = [f"## {recipe_name}\n"]
    
    md.append("### 🛒 Bahan-bahan")
    if ingredients:
        for ing in ingredients:
            md.append(f"- {ing}")
    else:
        md.append("- (Tidak ada daftar bahan terdeteksi dari teks)")
    
    md.append("\n### 🍳 Cara Memasak")
    if steps:
        for i, step in enumerate(steps, 1):
            md.append(f"{i}. {step}")
    elif remainder:
        for i, step in enumerate(remainder, 1):
            md.append(f"{i}. {step}")
    else:
        md.append("1. (Instruksi tidak tersedia)")
    
    md.append("\n### ℹ️ Informasi Nutrisi")
    nutrition_info = extract_nutrition_info(text)
    if nutrition_info:
        for info in nutrition_info:
            md.append(f"- {info}")
    else:
        md.append("- (Tidak ada informasi nutrisi terdeteksi)")
    
    result = "\n".join(md)
    print(f"🔧 LOCAL FORMATTER - Unstructured result:\n{result}")
    return result

def extract_nutrition_info(text: str) -> list:
    """Extract nutrition information from text."""
    nutrition_patterns = [
        r'Total Fat[^,\n]*',
        r'Saturated Fat[^,\n]*', 
        r'Cholesterol[^,\n]*',
        r'Sodium[^,\n]*',
        r'Total Carbohydrate[^,\n]*',
        r'Dietary Fiber[^,\n]*',
        r'Total Sugars[^,\n]*',
        r'Protein[^,\n]*',
        r'Vitamin[^,\n]*',
        r'Calcium[^,\n]*',
        r'Iron[^,\n]*',
        r'Potassium[^,\n]*'
    ]
    
    found_nutrition = []
    for pattern in nutrition_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        found_nutrition.extend(matches)
    
    return found_nutrition


# --- Format with Gemini (IMPROVED) ---
def format_with_gemini(raw_text: str, recipe_name: str) -> str:
    if not model_gemini:
        print("ℹ️ Gemini not configured — using local formatter.")
        return local_format_to_markdown(raw_text, recipe_name)

    print(f"🔍 GEMINI - Processing: {recipe_name}")
    
    # Clean text more carefully
    cleaned_text = raw_text.strip()
    if cleaned_text.lower().startswith(recipe_name.lower()):
        cleaned_text = cleaned_text[len(recipe_name):].lstrip(".:- ").strip()
        print(f"🧹 GEMINI - Cleaned redundant title")

    prompt = f"""
ANDA ADALAH AHLI RESEP PROFESIONAL. FORMAT TEKS BERIKUT MENJADI MARKDOWN YANG RAPI.

FORMAT YANG HARUS DIIKUTI:

## [NAMA RESEP]

### 🛒 Bahan-bahan
- [bahan 1]
- [bahan 2]

### 🍳 Cara Memasak  
1. [langkah 1]
2. [langkah 2]

### ℹ️ Informasi Nutrisi
- [info nutrisi 1]
- [info nutrisi 2]

ATURAN:
1. JUDUL HARUS: ## {recipe_name}
2. EKSTRAK semua bahan dan tempatkan di section Bahan-bahan
3. EKSTRAK semua langkah memasak dan tempatkan di section Cara Memasak  
4. EKSTRAK informasi nutrisi dan tempatkan di section Informasi Nutrisi
5. GUNAKAN format markdown di atas dengan TEPAT
6. JANGAN tambahkan konten lain selain format di atas

TEKS RESEP UNTUK DIFORMAT:

{cleaned_text}

HASIL MARKDOWN:
"""

    tried_models = ["gemini-pro", "gemini-1.5-flash", "gemini-1.0-pro"]
    for candidate_model in tried_models:
        try:
            print(f"🔍 GEMINI - Trying model: {candidate_model}")
            gen_model = genai.GenerativeModel(candidate_model)
            resp = gen_model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.1,
                    "max_output_tokens": 2048
                }
            )
            text = (resp.text or "").strip()

            print(f"📝 GEMINI - Raw response from {candidate_model}:")
            print("=" * 60)
            print(text)
            print("=" * 60)

            # Validate response
            is_valid = (
                text and 
                text.startswith(f"## {recipe_name}") and
                "### 🛒 Bahan-bahan" in text and
                "### 🍳 Cara Memasak" in text
            )
            
            if is_valid:
                print(f"✅ GEMINI - Success with {candidate_model}")
                return text
            else:
                print(f"❌ GEMINI - Invalid format from {candidate_model}")
                continue
                
        except Exception as e:
            print(f"🚨 GEMINI - Error with {candidate_model}: {e}")

    print("🔧 GEMINI - All models failed, falling back to local formatter")
    return local_format_to_markdown(raw_text, recipe_name)


# --- SEARCH + OUTPUT ---
def search_recipes(query: str, top_k: int = 3):
    try:
        print(f"🔍 SEARCH - Query: '{query}', top_k: {top_k}")
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
            score = point.score

            print(f"📊 SEARCH - Result {i+1}: {recipe_name} (score: {score:.3f})")
            
            # Format the result
            if i == 0:  # Only use Gemini for top result
                formatted_text = format_with_gemini(raw, recipe_name)
            else:
                formatted_text = local_format_to_markdown(raw, recipe_name)

            hits.append({
                "score": score,
                "text": formatted_text,
                "recipe_name": recipe_name,
            })

        print(f"✅ SEARCH - Found {len(hits)} results")
        return hits

    except Exception as e:
        print(f"🚨 SEARCH - Error: {e}")
        return [{"error": f"{type(e).__name__}: {e}"}]