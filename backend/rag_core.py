from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import re
import time

# spaCy (optional) - loaded lazily
_spacy_nlp = None
def get_spacy():
    global _spacy_nlp
    if _spacy_nlp is not None:
        return _spacy_nlp
    try:
        import spacy
        try:
            # try to load small English model
            _spacy_nlp = spacy.load("en_core_web_sm")
        except Exception:
            # fallback: load generic pipeline (may require download by user)
            _spacy_nlp = spacy.blank("en")
        return _spacy_nlp
    except Exception:
        _spacy_nlp = None
        return None

load_dotenv()

# --- GLOBAL PATTERNS ---
# Reusable compiled regexes and blacklists used across functions
UNIT_PATTERN = re.compile(r'\b(?:cup|cups|tbsp|tbs|tsp|tablespoon|tablespoons|teaspoon|teaspoons|gram|g|kg|ml|l|oz|ounce|ounces|pound|pounds|lb|lbs|slice|slices|clove|cloves|inch|buah|siung|biji|lembar|potong)\b', re.IGNORECASE)
TIME_TEMPERATURE_PATTERN = re.compile(r'\b(?:minute|minutes|min|hour|hours|hr|sec|second|seconds|degree|degrees|°|°c|°f|celsius|fahrenheit|oven|bake|preheat|roast|simmer|broil)\b', re.IGNORECASE)
NUTRITION_HINTS = re.compile(
    r'(%|vitamin\b|kcal\b|calorie|calories|kj\b|mg\b|per serving|serving|\bprotein\b|\bfat\b|\bsaturated fat\b|\bcholesterol\b|\bsodium\b|\bcarbohydrate\b|\bfiber\b|\bsugars?\b|\biron\b|\bpotassium\b|\bcalcium\b)',
    re.IGNORECASE
)
def extract_nutrition_info(text: str) -> list:
    """Extract nutrition-related lines from raw text as a separate list.
    Captures calories, vitamins, percentages, mg, per serving, etc.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    nutrition = []
    for ln in lines:
        # Direct nutrition matches
        if NUTRITION_HINTS.search(ln):
            nutrition.append(ln)
            continue
        # Common formats like: Calories: 250 per serving, Vitamin C 2%
        if re.search(r"\b(calories?|kcal|vitamin [a-z]|vitamin [A-Z]|\d+\s*mg|\d+\s*kJ|\d+\s*%)\b", ln, flags=re.IGNORECASE):
            nutrition.append(ln)
    # Deduplicate while preserving order
    seen = set()
    final = []
    for item in nutrition:
        key = item.lower()
        if key not in seen:
            final.append(item)
            seen.add(key)
    return final

VERB_PHRASES = re.compile(
    r"\b(peel|core|muddle|melt|invert|place|fold|mix|stir|add|combine|whisk|beat|pour|spread|slice|dice|chop|cut|press|roll|bake|cook|heat|boil|simmer|fry|roast|preheat|remove|brush|lay|set|serve|garnish)\b",
    re.IGNORECASE
)
KITCHEN_TOOLS = {
    'pan','pot','saucepan','bowl','sheet','baking sheet','plate','dish','spoon','fork','knife','glass',
    'cup','container','mixer','blender','skillet','oven','tray'
}

# --- CONFIG ---
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "recipes")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

# --- SETUP AI (Gemini) ---
genai_available = False
genai = None
try:
    import google.generativeai as genai
    
    if not hasattr(genai, 'GenerativeModel'):
        print("❌ LIBRARY USANG: Jalankan 'pip install -U google-generativeai'")
        genai = None
    elif GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        genai_available = True
        print("👉 Google Gemini API configured (Online).")
    else:
        print("⚠️ GOOGLE_API_KEY hilang. Mode Offline.")
        genai = None
        
except Exception as e:
    print(f"⚠️ Error setup Gemini: {e}")
    genai = None

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
        _client = QdrantClient(url=QDRANT_URL, prefer_grpc=False)
    return _client

def embed_text(text: str):
    return get_model().encode(text).tolist()

# ==========================================
# LANGUAGE DETECTION
# ==========================================
def detect_language(query: str) -> str:
    """Deteksi bahasa dari query user"""
    indonesian_keywords = [
        'resep', 'cara', 'membuat', 'memasak', 'bahan', 'apa', 'bagaimana',
        'dengan', 'untuk', 'yang', 'dan', 'adalah', 'ini', 'itu'
    ]
    
    query_lower = query.lower()
    indonesian_count = sum(1 for word in indonesian_keywords if word in query_lower)

    # simple heuristic: if more than 0 matches -> Indonesian, otherwise English
    if indonesian_count >= 1:
        return 'indonesian'
    # also detect by common Indonesian characters/words
    if re.search(r'\b(apa|tolong|tolonglah|silakan|saya|kamu|kue|nasi|goreng)\b', query_lower):
        return 'indonesian'

    # fallback: default to english
    return 'english'
    

# ==========================================
# TEXT CLEANER (PENTING: Hapus redundansi)
# ==========================================
def clean_garbage_text(text: str, title: str) -> str:
    """Membersihkan teks sampah sebelum diproses AI/Local."""
    cleaned = text.strip()

    # 1. Hapus Judul jika nempel di awal
    if cleaned.lower().startswith(title.lower()):
        cleaned = cleaned[len(title):].strip(" .:-")

    # 2. Hapus label section yang redundan
    pattern = r'^(Cara Memasak|Cara Membuat|Instructions|Steps|Method|Directions)[:\s\-]*'
    cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).strip()

    # 3. Hapus kredit di akhir
    cleaned = re.sub(r'(Dotdash Meredith|Allrecipes|Food Studios).*$', '', cleaned, flags=re.IGNORECASE).strip()
    
    # 4. Hapus duplikasi baris
    lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
    unique_lines = []
    seen = set()
    for line in lines:
        line_clean = re.sub(r'^\d+[\.\)]\s*', '', line.lower())
        if line_clean not in seen:
            unique_lines.append(line)
            seen.add(line_clean)
    
    return '\n'.join(unique_lines)

# ==========================================
# INGREDIENT EXTRACTION (IMPROVED)
# ==========================================
def extract_ingredients_from_text(text: str) -> list:
    """Extract bahan dari raw text dengan pattern matching yang lebih baik"""
    ingredients = []

    # Units that are likely to be ingredients (expand as needed)
    units = r'(?:cup|cups|tbsp|tbs|tsp|tablespoon|tablespoons|teaspoon|teaspoons|gram|g|kg|ml|l|oz|ounce|ounces|pound|pounds|lb|lbs|buah|siung|biji|lembar|potong|slices|slice|clove|cloves)'

    # Pattern untuk mendeteksi bahan yang jelas (angka + unit + nama bahan)
    patterns = [
        rf'(\d+(?:[\.,]\d+)?\s*{units}\s+[^.,\n]+)',
        rf'(\d+[-–]\d+\s*{units}\s+[^.,\n]+)',
        rf'(\d+\s+to\s+\d+\s*{units}\s+[^.,\n]+)',
        rf'(\d+/\d+\s*{units}\s+[^.,\n]+)',
        # pola lain: unit bisa muncul di tengah kalimat (mis. "add 2 cups flour to the mix")
        rf'([\w\s\,\(\)\-]*\d+(?:[\.,]\d+)?\s*{units}\s*(?:\([^\)]*\)\s*)?[^.,\n]+)',
        # pola untuk item dengan ukuran di dalam tanda kurung: "1 (9-inch) pie crust"
        rf'(\d+\s*\([^\)]+\)\s*[^.,\n]+)'
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            # Skip nutrition-like lines (percentages, vitamin info, kcal, mg, etc.)
            if NUTRITION_HINTS.search(m):
                continue
            # Skip explicit time/temperature matches
            if TIME_TEMPERATURE_PATTERN.search(m):
                continue
            # Skip lines that start with obvious cooking verbs
            if VERB_PHRASES.match(m.strip()):
                continue
            ingredients.append(m)

    # Post-split matches on newlines/bullets to avoid merged multi-line captures
    split_candidates = []
    for it in ingredients:
        for part in re.split(r'\r?\n|\n|\n-|-\s+', it):
            p = part.strip().lstrip('-').strip()
            if not p:
                continue
            # Skip short, nutrition, or temperature-like parts
            if NUTRITION_HINTS.search(p) or TIME_TEMPERATURE_PATTERN.search(p):
                continue
            # Filter out kitchen tools present in phrase
            if any(tool in p.lower() for tool in KITCHEN_TOOLS):
                continue
            # Skip if phrase begins with a cooking verb
            if VERB_PHRASES.match(p):
                continue
            split_candidates.append(p)

    # replace ingredients list with cleaned split candidates
    ingredients = split_candidates

    # Additional heuristics and blacklist are applied below using compiled patterns

    cleaned_ingredients = []
    seen = set()
    for ing in ingredients:
        ing_clean = ing.strip()
        ing_lower = ing_clean.lower()

        # Skip obvious time/temperature matches
        if TIME_TEMPERATURE_PATTERN.search(ing_clean):
            continue

        # Skip nutrition or calorie hints inside candidate
        if NUTRITION_HINTS.search(ing_clean):
            continue

        # Skip if candidate mentions kitchen tools
        if any(tool in ing_clean.lower() for tool in KITCHEN_TOOLS):
            continue

        # Skip verb-led phrases (stricter)
        if VERB_PHRASES.match(ing_clean):
            continue

        # Skip short garbage matches
        if len(ing_clean) < 3:
            continue

        # Deduplicate
        if ing_lower not in seen:
            cleaned_ingredients.append(ing_clean)
            seen.add(ing_lower)

    # Jika tidak ditemukan bahan eksplisit, coba heuristik lain: cari kata setelah kata kerja masak yang umum
    if not cleaned_ingredients:
        # First: try spaCy noun-chunk extraction (preferred, more accurate)
        nlp = get_spacy()
        heuristics = []
        if nlp is not None:
            try:
                doc = nlp(text)
                # collect candidate noun chunks and nouns
                for chunk in doc.noun_chunks:
                    cand = chunk.text.strip()
                    # filter trivial chunks
                    if len(cand) < 2 or re.search(r'\d', cand):
                        continue
                    # ignore very long chunks
                    if len(cand.split()) > 5:
                        continue
                    # reject nutrition/time chunks
                    if NUTRITION_HINTS.search(cand) or TIME_TEMPERATURE_PATTERN.search(cand):
                        continue
                    heuristics.append(cand)

                # also pick single nouns that look relevant
                for token in doc:
                    if token.pos_ in ('NOUN', 'PROPN') and token.is_alpha:
                        tok = token.text.strip()
                        if NUTRITION_HINTS.search(tok) or TIME_TEMPERATURE_PATTERN.search(tok):
                            continue
                        heuristics.append(tok)
            except Exception:
                heuristics = []

        # If spaCy not available or returned nothing useful, fallback to verb-based heuristics
        if not heuristics:
            verb_pattern = r"\b(?:add|adds|stir in|stir|mix in|mix|combine|fold in|fold|sprinkle|melt|press|place|roll out|roll|peel|core|slice|top with|use|put|brush)\b\s*(?:the\s)?([^\.,;\n]+)"
            for match in re.findall(verb_pattern, text, flags=re.IGNORECASE):
                cand = match.strip()
                parts = re.split(r",| and |\band\b|;|/|\+|\bor\b", cand, flags=re.IGNORECASE)
                for part in parts:
                    part = part.strip()
                    part = re.sub(r"\binto\b.*$", '', part, flags=re.IGNORECASE).strip()
                    part = re.sub(r"\bfor\b.*$", '', part, flags=re.IGNORECASE).strip()
                    words = re.findall(r"[A-Za-z\-']+", part)
                    stopwords = set(['and','then','the','a','an','to','in','into','over','with','until','for','about','of','on','at','by','each','per','so','it','that'])
                    cooking_adj = set(['sliced','thinly','chopped','diced','minced','fresh','unsalted','salted','medium','large','small','peeled','cored','thin','soft','remaining','one','two','both','stir','form','first','second','third','fourth','up','will','trim','excess','some','onto','sides','inch','strip','strips','pieces','piece','press','roll','rollout','lay','cut','place','cook','bring','reduce','remove','paste'])
                    keep = [w for w in words if w.lower() not in stopwords and w.lower() not in cooking_adj and not re.match(r'^\d', w)]
                    if not keep:
                        continue
                    candidate = keep[0].strip()
                    # Reject if candidate is purely a unit/measurement or a nutrition hint
                    if len(candidate) > 1 and candidate.lower() not in seen:
                        if UNIT_PATTERN.search(candidate) and not re.search(r'[A-Za-z]{3,}', re.sub(UNIT_PATTERN.pattern, '', candidate, flags=re.IGNORECASE)):
                            # candidate contains only units/measurements -> skip
                            continue
                        if NUTRITION_HINTS.search(candidate) or TIME_TEMPERATURE_PATTERN.search(candidate):
                            continue
                        if any(tool in candidate.lower() for tool in KITCHEN_TOOLS):
                            continue
                        heuristics.append(candidate)

        # Dedup heuristics and add to cleaned_ingredients
        for h in heuristics:
            h_clean = h.strip()
            if h_clean and h_clean.lower() not in seen:
                cleaned_ingredients.append(h_clean)
                seen.add(h_clean.lower())

    # -----------------------------
    # Post-process & filter noisy candidates
    # -----------------------------
    # Lowercase stoplist of obvious non-ingredient words/phrases
    banned_substrings = {
        'oven', 'preheat', 'minute', 'minutes', 'hour', 'heat', 'step', 'steps', 'repeat', 'natalie',
        'informasi', 'nutrisi', 'instructions', 'cara', 'memasak', 'remove', 'place', 'lay', 'fold',
        'unfold', 'trim', 'press', 'bake', 'cook', 'bring', 'reduce', 'repeat', 'position', 'center', 'edge',
        'vitamin', 'kcal', 'calorie', '%', 'serving', 'per serving'
    }

    # small pantry/common ingredient list to prefer matches
    pantry = {
        'salt','sugar','butter','flour','egg','eggs','milk','water','olive oil','oil','vinegar',
        'baking powder','baking soda','yeast','vanilla','cinnamon','apple','apples','pastry','crust',
        'sugar-butter','sugar butter','sugars'
    }

    # reuse module-level UNIT_PATTERN
    unit_pattern = UNIT_PATTERN

    # Consolidate by core ingredient name to avoid duplicate entries like
    # '1/2 cup butter' and bogus '2 cup butter' produced by noisy matches.
    core_map = {}
    core_order = []

    def core_name(s: str) -> str:
        s0 = s.lower()
        s0 = re.sub(r'\([^)]*\)', ' ', s0)  # drop parentheses
        s0 = re.sub(r'[^a-z\s]', ' ', s0)
        s0 = re.sub(UNIT_PATTERN.pattern, ' ', s0, flags=re.IGNORECASE)
        s0 = re.sub(r'\b(?:sliced|chopped|diced|minced|fresh|unsalted|salted|medium|large|small|peeled|cored|thin|soft|pieces|piece)\b',' ', s0)
        s0 = re.sub(r'\s+', ' ', s0).strip()
        parts = s0.split()
        if not parts:
            return s0
        return ' '.join(parts[-2:]) if len(parts) >= 2 else parts[-1]

    for cand in cleaned_ingredients:
        c = cand.strip()
        c = re.sub(r'^[^A-Za-z0-9]+|[^A-Za-z0-9]+$', '', c)
        if not c:
            continue
        cl = c.lower()

        if any(bs in cl for bs in banned_substrings):
            continue

        if re.search(r'\b(?:add|stir|mix|combine|fold|press|cut|roll|preheat|bake|cook|bring|reduce|remove|brush)\b', cl):
            if not (unit_pattern.search(cl) or any(p in cl for p in pantry)):
                continue

        # Hard filter: remove kitchen tools-only lines
        if any(tool in cl for tool in KITCHEN_TOOLS):
            # allow only if also contains pantry ingredient word
            if not any(p in cl for p in pantry):
                continue

        if len(re.sub(r'[^A-Za-z]+', '', cl)) < 3:
            continue

        cl = cl.replace('sugars', 'sugar')
        cl = cl.replace('apples', 'apple')

        core = core_name(c)
        if core not in core_map:
            core_map[core] = c
            core_order.append(core)
        else:
            existing = core_map[core]
            # prefer candidate with explicit measurement (digit) or longer description
            if re.search(r'\d', c) and not re.search(r'\d', existing):
                core_map[core] = c
            elif len(c) > len(existing):
                core_map[core] = c

    final_ingredients = [core_map[k] for k in core_order]

    # If we found high-confidence final ingredients, return them; otherwise fall back to previous cleaned list
    if final_ingredients:
        return final_ingredients

    return cleaned_ingredients

# ==========================================
# LOCAL FORMATTER (Offline Fallback)
# ==========================================
def local_format_to_markdown(raw_text: str, recipe_name: str, language: str = "english", ingredients_list=None) -> str:
    print(f"🔧 LOCAL FORMATTER (Offline): {recipe_name}")

    # Bersihkan dulu!
    text = clean_garbage_text(raw_text, recipe_name)

    # Jika payload sudah menyertakan daftar bahan, pakai itu. Kalau tidak, coba ekstrak secara heuristik
    if ingredients_list:
        if isinstance(ingredients_list, list):
            ingredients = [str(x).strip() for x in ingredients_list if str(x).strip()]
        else:
            ingredients = [ln.strip() for ln in str(ingredients_list).split('\n') if ln.strip()]
    else:
        ingredients = extract_ingredients_from_text(text)
    
    # Extract nutrition info BEFORE filtering it out from steps
    nutrition_info = extract_nutrition_info(text)

    # Remove explicit ingredient lines from the text before splitting steps
    text_for_steps = text
    if ingredients:
        # Remove header like 'Ingredients' or 'Bahan-bahan'
        text_for_steps = re.sub(r'(?im)^\s*(bahan-bahan|ingredients|ingredient)[:\s\-]*\n?', '', text_for_steps)
        # Remove each ingredient line (use escaped exact-match removal)
        for ing in ingredients:
            esc = re.escape(ing)
            text_for_steps = re.sub(rf'(?im)^\s*(?:-|\u2022)?\s*{esc}\s*$', '', text_for_steps)
        # collapse multiple blank lines
        text_for_steps = re.sub(r'\n{2,}', '\n\n', text_for_steps).strip()
        # Remove nutrition/info lines from steps text
        text_for_steps = '\n'.join([ln for ln in text_for_steps.splitlines() if not NUTRITION_HINTS.search(ln)])
        text_for_steps = text_for_steps.strip()

    # Split menjadi steps
    sentences = re.split(r'(?<=[.!?])\s+', text_for_steps)
    steps = []

    for s in sentences:
        s = s.strip()
        if len(s) < 10: 
            continue
        
        # Skip if this sentence is likely an ingredient-only line:
        # - short (<=8 words) and contains a unit, OR exactly matches an extracted ingredient
        short_words = len(s.split()) <= 8
        contains_unit = UNIT_PATTERN.search(s) is not None
        matches_exact_ing = any(s.lower().strip().startswith(ing.lower()) or s.lower().strip() == ing.lower() for ing in ingredients)
        if (short_words and contains_unit) or matches_exact_ing:
            continue
            
        # Hapus penomoran ganda
        s = re.sub(r'^\d+[\.\)]\s*', '', s)
        
        if s:
            steps.append(s)

    # Format berdasarkan bahasa
    if language == "indonesian":
        md = [f"## {recipe_name}\n"]
        md.append("### 🛒 Bahan-bahan")
        
        if ingredients:
            for ing in ingredients:
                md.append(f"- {ing}")
        else:
            md.append("- (Bahan tidak terdeteksi, lihat instruksi di bawah)")
        
        md.append("\n### 🍳 Cara Memasak")
        if steps:
            for i, step in enumerate(steps, 1):
                md.append(f"{i}. {step}")
        else:
            md.append(f"1. {text}")

        # Nutrition section (if available)
        md.append("\n### ℹ️ Informasi Nutrisi")
        if nutrition_info:
            for item in nutrition_info:
                md.append(f"- {item}")
        else:
            md.append("- Tidak tersedia")
    else:
        md = [f"## {recipe_name}\n"]
        md.append("### 🛒 Ingredients")
        
        if ingredients:
            for ing in ingredients:
                md.append(f"- {ing}")
        else:
            md.append("- (Ingredients not auto-detected, see instructions)")
        
        md.append("\n### 🍳 Instructions")
        if steps:
            for i, step in enumerate(steps, 1):
                md.append(f"{i}. {step}")
        else:
            md.append(f"1. {text}")

        # Nutrition section (if available)
        md.append("\n### ℹ️ Nutrition Information")
        if nutrition_info:
            for item in nutrition_info:
                md.append(f"- {item}")
        else:
            md.append("- Not available")
    
    return "\n".join(md)


def format_strict_from_payload(payload: dict, language: str = "indonesian") -> str:
    """Format output strictly without mixing sections using payload fields only.

    Structure:
    # Nama Resep
    ## Ingredients
    - ...
    ## Instruction
    - ... (each step split by ';')
    ## Nutrition
    """
    name = payload.get("recipe_name", "Tanpa Judul")
    # Prefer steps array; fallback to steps_text or extract from text
    steps_raw = payload.get("steps")
    steps_text = payload.get("steps_text")
    raw_text = payload.get("text", "")
    nutrients_text = payload.get("nutrients_text")

    # Build instruction steps as list split by ';'
    if isinstance(steps_raw, list) and steps_raw:
        joined = "; ".join([str(s).strip() for s in steps_raw if str(s).strip()])
        instr_list = [seg.strip() for seg in joined.split(";") if seg.strip()]
    elif isinstance(steps_text, str) and steps_text.strip():
        instr_list = [seg.strip() for seg in steps_text.split(";") if seg.strip()]
    else:
        # Fallback: extract from raw_text
        instr_list = extract_steps(raw_text, name)

    # Ingredients inferred from steps text (not from nutrients/text)
    ing_source_text = "; ".join(instr_list)
    ingredients = extract_ingredients_from_text(ing_source_text)

    # Nutrition section: prefer nutrients_text; else extract
    if isinstance(nutrients_text, str) and nutrients_text.strip():
        # split by commas into items
        nutrition_items = [i.strip() for i in nutrients_text.split(",") if i.strip()]
    else:
        nutrition_items = extract_nutrition_info(raw_text)

    # Assemble markdown in requested language, with strict section separation
    if language == "indonesian":
        md = [f"# {name}", "---", "## Ingridients"]
        if ingredients:
            md += [f"- {ing}" for ing in ingredients]
        else:
            md.append("- (Bahan tidak tersedia)")
        md.append("## Instruction")
        if instr_list:
            md += [f"- {st}" for st in instr_list]
        else:
            md.append("- (Langkah tidak tersedia)")
        md.append("## Nutrition")
        if nutrition_items:
            md += [f"- {n}" for n in nutrition_items]
        else:
            md.append("- Tidak tersedia")
    else:
        md = [f"# {name}", "---", "## Ingredients"]
        if ingredients:
            md += [f"- {ing}" for ing in ingredients]
        else:
            md.append("- (Ingredients not available)")
        md.append("## Instruction")
        if instr_list:
            md += [f"- {st}" for st in instr_list]
        else:
            md.append("- (Steps not available)")
        md.append("## Nutrition")
        if nutrition_items:
            md += [f"- {n}" for n in nutrition_items]
        else:
            md.append("- Not available")

    return "\n".join(md)


def extract_steps(raw_text: str, recipe_name: str, ingredients_list=None) -> list:
    """Return cleaned list of instruction steps (used for structured responses)."""
    text = clean_garbage_text(raw_text, recipe_name)

    # If ingredients_list provided, remove their explicit lines from text.
    # If not provided, attempt to extract ingredients and remove their lines to avoid mixing.
    if ingredients_list:
        if isinstance(ingredients_list, list):
            ings = [str(x).strip() for x in ingredients_list if str(x).strip()]
        else:
            ings = [ln.strip() for ln in str(ingredients_list).split('\n') if ln.strip()]
    else:
        ings = extract_ingredients_from_text(text)

    for ing in ings:
        esc = re.escape(ing)
        # allow optional bullet markers (- or •) and surrounding whitespace
        text = re.sub(rf'(?im)^\s*(?:-|\u2022)?\s*{esc}\s*$', '', text)

    # Split into sentences/steps
    sentences = re.split(r'(?<=[.!?])\s+', text)
    steps = []
    for s in sentences:
        s = s.strip()
        if len(s) < 6:
            continue
        # remove numbering
        s = re.sub(r'^\d+[\.\)]\s*', '', s)
        # skip if looks like an ingredient-only line (short + unit)
        if (len(s.split()) <= 6 and UNIT_PATTERN.search(s)):
            continue
        # also skip nutrition lines; they are handled by extract_nutrition_info
        if NUTRITION_HINTS.search(s):
            continue
        steps.append(s)
    # deduplicate preserving order
    seen = set()
    final = []
    for st in steps:
        key = st.lower()
        if key not in seen:
            final.append(st)
            seen.add(key)
    return final


# ==========================================
# MAIN FORMAT FUNCTION - CONSISTENT OUTPUT
# ==========================================
def format_recipe_output(payload: dict, language: str = "indonesian") -> str:
    """
    Format resep dengan struktur yang konsisten dan TIDAK TERCAMPUR:
    
    # Nama Resep
    ---
    ## Ingredients
    - bahan 1
    - bahan 2
    ## Instruction
    - langkah 1
    - langkah 2
    ## Nutrition
    - info nutrisi
    
    Args:
        payload: Dictionary dari Qdrant dengan keys: recipe_name, steps, nutrients
        language: "indonesian" atau "english"
    """
    name = payload.get("recipe_name", "Tanpa Judul")
    
    # 1. AMBIL STEPS (dari payload.steps yang sudah berbentuk list)
    steps = payload.get("steps", [])
    if isinstance(steps, list) and steps:
        # Jika steps adalah list of strings, split per kalimat untuk lebih readable
        instruction_list = []
        for step in steps:
            if isinstance(step, str):
                # Split long steps into sentences
                sentences = re.split(r'(?<=[.!?])\s+', step.strip())
                for sent in sentences:
                    sent = sent.strip()
                    # Remove author credits and photo credits
                    sent = re.sub(r'\b[A-Z][a-z]+\s+[A-Z]\s*$', '', sent).strip()
                    sent = re.sub(r'^Photo by\s+.*$', '', sent, flags=re.IGNORECASE).strip()
                    sent = re.sub(r'^Recipe by\s+.*$', '', sent, flags=re.IGNORECASE).strip()
                    sent = re.sub(r"cookin['\']?\s*mama", '', sent, flags=re.IGNORECASE).strip()
                    if len(sent) > 15:  # Only keep meaningful sentences
                        instruction_list.append(sent)
    elif isinstance(payload.get("steps_text"), str):
        # Fallback ke steps_text jika ada
        steps_text = payload.get("steps_text", "")
        instruction_list = []
        for step in steps_text.split(";"):
            sentences = re.split(r'(?<=[.!?])\s+', step.strip())
            for sent in sentences:
                sent = sent.strip()
                sent = re.sub(r'\b[A-Z][a-z]+\s+[A-Z]\s*$', '', sent).strip()
                if len(sent) > 15:
                    instruction_list.append(sent)
    else:
        # Last fallback: extract from raw text
        raw_text = payload.get("text", "")
        instruction_list = extract_steps(raw_text, name)
    
    # 2. EKSTRAK INGREDIENTS dari instruction text dengan pendekatan berbeda
    # Karena ingredients ada di dalam instructions, kita parse dari kata kerja
    ingredients = []
    ingredients_set = set()
    
    # Kata kerja yang biasanya diikuti dengan bahan
    cooking_verbs = r'\b(add|melt|mix|stir|combine|beat|whisk|fold|pour|peel|core|slice|dice|chop|cut|spread|sprinkle|brush|use|place|layer)\b'
    
    # Kata-kata junk yang harus difilter
    junk_words = {'first', 'second', 'third', 'fourth', 'one', 'two', 'three', 'four', 'both', 
                  'remaining', 'unused', 'some', 'each', 'all', 'other', 'another', 'more',
                  'oven', 'pan', 'bowl', 'sheet', 'heat', 'temperature', 'degrees', 'strips',
                  'crust', 'lattice', 'mound', 'pieces', 'slices', 'mixture', 'position'}
    
    for step in instruction_list:
        # Pattern khusus untuk "Combine X, Y, and Z" atau "Mix A, B, C"
        combine_match = re.match(r'^(combine|mix)\s+([^;]+?)(?:\s+in\s+|\s*;)', step, re.IGNORECASE)
        if combine_match:
            items_text = combine_match.group(2)
            # Split by "and" dan ","
            parts = re.split(r',\s*(?:and\s+)?|\s+and\s+', items_text)
            for part in parts:
                part = part.strip()
                part = re.sub(r'\b(the|a|an|some|all|both|remaining)\b\s*', '', part, flags=re.IGNORECASE)
                part = re.sub(r'\s+', ' ', part).strip()
                
                if len(part) < 3:
                    continue
                words = part.lower().split()
                if len(words) == 1 and words[0] in junk_words:
                    continue
                food_words = [w for w in words if len(w) >= 3 and w not in junk_words]
                if not food_words:
                    continue
                    
                part_lower = part.lower()
                if part_lower not in ingredients_set:
                    ingredients.append(part)
                    ingredients_set.add(part_lower)
            continue
        
        # Cari pola: kata kerja + bahan
        # Contoh: "Melt butter", "Add flour and sugar", "Peel and core apples"
        matches = re.finditer(rf'{cooking_verbs}\s+([^,.;]+?)(?:\s+(?:in|into|over|on|to|until|and\s+(?:stir|cook|bring)|,|\.))', step, re.IGNORECASE)
        for match in matches:
            ingredient_phrase = match.group(2).strip()
            # Clean up
            ingredient_phrase = re.sub(r'\b(the|a|an|some|all|both|remaining)\b\s*', '', ingredient_phrase, flags=re.IGNORECASE)
            ingredient_phrase = re.sub(r'\s+', ' ', ingredient_phrase).strip()
            
            # Filter out junk
            if len(ingredient_phrase) < 3:
                continue
            
            # Skip single junk words
            if len(ingredient_phrase.split()) == 1 and ingredient_phrase.lower() in junk_words:
                continue
                
            # Skip phrases that are primarily junk words
            words = ingredient_phrase.lower().split()
            if all(w in junk_words for w in words):
                continue
            
            # Must contain at least one food-like word (3+ letters, not a junk word)
            food_words = [w for w in words if len(w) >= 3 and w not in junk_words]
            if not food_words:
                continue
                
            ing_lower = ingredient_phrase.lower()
            if ing_lower not in ingredients_set:
                ingredients.append(ingredient_phrase)
                ingredients_set.add(ing_lower)
    
    # 3. AMBIL NUTRITION (dari payload.nutrients yang sudah berbentuk list)
    nutrients = payload.get("nutrients", [])
    if isinstance(nutrients, list) and nutrients:
        nutrition_list = [str(n).strip() for n in nutrients if str(n).strip()]
    elif isinstance(payload.get("nutrients_text"), str):
        # Fallback ke nutrients_text jika masih format lama
        nutrients_text = payload.get("nutrients_text", "")
        nutrition_list = [n.strip() for n in nutrients_text.split(",") if n.strip()]
    else:
        # Extract from raw text
        raw_text = payload.get("text", "")
        nutrition_list = extract_nutrition_info(raw_text)
    
    # 4. FORMAT OUTPUT DENGAN STRICT SEPARATION
    if language == "indonesian":
        output = [
            f"# {name}",
            "---",
            "## Ingredients"
        ]
        
        if ingredients:
            for ing in ingredients:
                output.append(f"- {ing}")
        else:
            output.append("- (Bahan tidak terdeteksi)")
        
        output.append("## Instruction")
        if instruction_list:
            for step in instruction_list:
                output.append(f"- {step}")
        else:
            output.append("- (Langkah tidak tersedia)")
        
        output.append("## Nutrition")
        if nutrition_list:
            for nut in nutrition_list:
                output.append(f"- {nut}")
        else:
            output.append("- Tidak tersedia")
    else:
        output = [
            f"# {name}",
            "---",
            "## Ingredients"
        ]
        
        if ingredients:
            for ing in ingredients:
                output.append(f"- {ing}")
        else:
            output.append("- (Ingredients not detected)")
        
        output.append("## Instruction")
        if instruction_list:
            for step in instruction_list:
                output.append(f"- {step}")
        else:
            output.append("- (Steps not available)")
        
        output.append("## Nutrition")
        if nutrition_list:
            for nut in nutrition_list:
                output.append(f"- {nut}")
        else:
            output.append("- Not available")
    
    return "\n".join(output)

    
def format_with_gemini(raw_text: str, recipe_name: str, user_query: str, ingredients_list=None) -> str:
    # Always detect language from query; prefer Indonesian when detected
    language = detect_language(user_query)
    if not genai_available:
        return local_format_to_markdown(raw_text, recipe_name, language)

    print(f"🔍 GEMINI Processing: {recipe_name}")

    # Bersihkan teks dulu
    cleaned_text = clean_garbage_text(raw_text, recipe_name)
    
    # Deteksi bahasa (already determined above)

    # Jika ada ingredients_list, tambahkan ke RAW_DATA agar Gemini memakai daftar bahan yang benar
    ingredients_block = ""
    if ingredients_list:
        if isinstance(ingredients_list, list):
            ingredients_block = "\n".join([f"- {i}" for i in ingredients_list])
        else:
            ingredients_block = str(ingredients_list)

    prompt = f"""
ROLE: Professional Chef Assistant

CRITICAL RULES:
1. **LANGUAGE STRICTNESS**: User query language: {"INDONESIAN" if language == "indonesian" else "ENGLISH"}
    - Respond ONLY in this language (STRICT).
    - Localize ALL headers and content accordingly.

2. **INGREDIENTS EXTRACTION**:
    - Extract ONLY ingredient nouns/phrases with measurements (e.g., "1 cup sugar", "500 g flour").
    - DO NOT include verbs (e.g., "peel", "melt"), actions, or tools (e.g., "saucepan", "baking sheet").
    - Normalize plural to singular where appropriate (e.g., "apples" → "apple").
    - If explicit list is missing, infer from text but keep STRICT filtering.

3. **INSTRUCTIONS FORMATTING**:
   - Create clear, numbered steps (1., 2., 3.)
   - Remove redundant labels like "Instructions:", "Cara Memasak:"
   - Each step should be one clear action
   - Remove duplicates

4. **NO REDUNDANCY**:
   - Never repeat section titles
   - Never duplicate steps
   - Keep it clean and concise

5. **STRICTNESS**:
    - Temperature low; avoid creative paraphrasing.
    - Prefer shortest correct phrasing.

USER QUERY: "{user_query}"
RECIPE NAME: "{recipe_name}"
RAW DATA: "{cleaned_text}"
RAW INGREDIENTS: "{ingredients_block}"

OUTPUT FORMAT ({"INDONESIAN" if language == "indonesian" else "ENGLISH"}):

## {recipe_name}

### 🛒 {"Bahan-bahan" if language == "indonesian" else "Ingredients"}
- [List all ingredients with measurements]

### 🍳 {"Cara Memasak" if language == "indonesian" else "Instructions"}
1. [First step]
2. [Second step]
...

### ℹ️ {"Informasi Nutrisi" if language == "indonesian" else "Nutrition Information"}
- [If available, otherwise write "{"Tidak tersedia" if language == "indonesian" else "Not available"}"]

IMPORTANT: Respond ONLY in {"INDONESIAN" if language == "indonesian" else "ENGLISH"}!
"""

    # Prefer more capable model first; use low temperature for strictness
    tried_models = ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro"]

    for model_name in tried_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt,
                generation_config={"temperature": 0.0, "max_output_tokens": 1600}
            )
            result_text = response.text.strip()
            
            if "##" in result_text:
                print(f"✅ GEMINI Success ({model_name})")
                # Post-process using our strict local formatter to ensure cleanliness
                return local_format_to_markdown(raw_text, recipe_name, language, ingredients_list=ingredients_list)
        except Exception as e:
            print(f"⚠️ {model_name} failed: {e}")
            continue
    
    # Fallback to local formatter if all Gemini models fail
    print("⚠️ All Gemini models failed, falling back to local formatter")
    return local_format_to_markdown(raw_text, recipe_name, language, ingredients_list=ingredients_list)

def llm_extract_structured(raw_text: str, recipe_name: str, target_language: str = 'indonesian') -> dict:
    """Use LLM (Gemini) to extract structured ingredients + instructions and return translations.

    Returns dict with keys: ingredients (list), instructions (list), text_id (markdown Indonesian)
    Falls back to local heuristics if Gemini not available or fails.
    """
    # Fallback result using local extraction
    fallback = {}
    try:
        fallback_ings = extract_ingredients_from_text(raw_text)
        fallback_steps = extract_steps(raw_text, recipe_name)
        fallback_text_id = local_format_to_markdown(raw_text, recipe_name, language='indonesian', ingredients_list=fallback_ings)
        fallback = {
            'ingredients': fallback_ings,
            'instructions': fallback_steps,
            'text_id': fallback_text_id
        }
    except Exception:
        fallback = {'ingredients': [], 'instructions': [], 'text_id': ''}

    if not genai_available or genai is None:
        return fallback
        fallback = {'ingredients': [], 'instructions': [], 'text_id': ''}

    if not genai_available:
        return fallback

    # Build a JSON-only prompt asking Gemini to return structured JSON
    cleaned = clean_garbage_text(raw_text, recipe_name)
    prompt = f"""
    You are a helpful assistant that extracts recipe data.

    INPUT_RECIPE_NAME: {recipe_name}
    INPUT_TEXT:
    {cleaned}

    TASK: Extract two lists: 'ingredients' (each item short, include measurement if present) and 'instructions' (numbered steps as short sentences). Also produce an Indonesian markdown version of the recipe with headers 'Bahan-bahan' and 'Cara Memasak'.

    REPLY ONLY AS JSON with keys: ingredients, instructions, text_id
    Example:
    {{"ingredients": ["1 cup sugar", "2 eggs"], "instructions": ["Preheat oven...","Mix..."] , "text_id":"## ..."}}
    Respond ONLY with valid JSON.
    """

    tried = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
    for model_name in tried:
        try:
            model = genai.GenerativeModel(model_name)
            resp = model.generate_content(prompt, generation_config={"temperature": 0.0, "max_output_tokens": 1200})
            txt = resp.text.strip()
            # try to find JSON substring
            import json as _json
            try:
                parsed = _json.loads(txt)
                # normalize keys
                ingredients = parsed.get('ingredients') or parsed.get('ing') or []
                instructions = parsed.get('instructions') or parsed.get('steps') or []
                text_id = parsed.get('text_id') or parsed.get('indonesian_markdown') or ''

                # Post-process: normalize ingredient strings
                def normalize_ingredient(s: str) -> str:
                    s0 = s.strip()
                    # remove enclosing punctuation
                    s0 = re.sub(r'^[^A-Za-z0-9]+|[^A-Za-z0-9]+$', '', s0)
                    s0 = s0.replace('\t', ' ').replace('\n', ' ').strip()
                    # common unit normalization
                    s0 = re.sub(r'\bTablespoons?\b', 'tbsp', s0, flags=re.IGNORECASE)
                    s0 = re.sub(r'\bTablespoon\b', 'tbsp', s0, flags=re.IGNORECASE)
                    s0 = re.sub(r'\bTeaspoons?\b', 'tsp', s0, flags=re.IGNORECASE)
                    s0 = re.sub(r'\bGrams?\b', 'g', s0, flags=re.IGNORECASE)
                    s0 = re.sub(r'\bKilograms?\b', 'kg', s0, flags=re.IGNORECASE)
                    s0 = re.sub(r'\bMilliliters?\b', 'ml', s0, flags=re.IGNORECASE)
                    s0 = re.sub(r'\bCup(s)?\b', 'cup', s0, flags=re.IGNORECASE)
                    # collapse multiple spaces
                    s0 = re.sub(r'\s{2,}', ' ', s0)
                    return s0

                norm_ings = []
                seen_ing = set()
                if isinstance(ingredients, list):
                    for ii in ingredients:
                        try:
                            s = str(ii)
                        except Exception:
                            continue
                        n = normalize_ingredient(s)
                        if len(re.sub(r'[^A-Za-z]+','', n)) < 2:
                            continue
                        nl = n.lower()
                        if nl in seen_ing:
                            continue
                        seen_ing.add(nl)
                        norm_ings.append(n)

                # normalize instructions to short sentences
                norm_instr = []
                if isinstance(instructions, list):
                    for st in instructions:
                        stt = str(st).strip()
                        stt = re.sub(r'\s+', ' ', stt)
                        # remove trailing punctuation gaps
                        stt = stt.strip()
                        if len(stt) < 6:
                            continue
                        norm_instr.append(stt)

                return {
                    'ingredients': norm_ings,
                    'instructions': norm_instr,
                    'text_id': text_id
                }
            except Exception:
                # If not valid JSON, try to extract JSON-like block
                import re as _re
                m = _re.search(r'\{[\s\S]*\}', txt)
                if m:
                    try:
                        parsed = _json.loads(m.group(0))
                        ingredients = parsed.get('ingredients') or []
                        instructions = parsed.get('instructions') or []
                        text_id = parsed.get('text_id') or ''
                        return {'ingredients': ingredients, 'instructions': instructions, 'text_id': text_id}
                    except Exception:
                        pass
                # else fallback to local
                return fallback
        except Exception:
            continue

    return fallback

# ==========================================
# SEARCH FUNCTION WITH ERROR HANDLING
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

        points = search_result.points

        # ==========================================
        # ERROR HANDLING: No Results Found
        # ==========================================
        if not points or len(points) == 0:
            language = detect_language(query)
            
            if language == "indonesian":
                return [{
                    "recipe_name": "Tidak Ditemukan",
                    "text": "## Maaf, Resep Tidak Ditemukan 😔\n\n"
                           "Resep yang Anda cari tidak ada di database kami.\n\n"
                           "### 💡 Saran:\n"
                           "- Coba kata kunci yang berbeda\n"
                           "- Periksa ejaan resep\n"
                           "- Cari resep serupa yang mungkin tersedia\n\n"
                           "**Mau saya rekomendasikan resep populer lainnya?** 🍽️",
                    "score": 0.0,
                    "not_found": True
                }]
            else:
                return [{
                    "recipe_name": "Not Found",
                    "text": "## Sorry, Recipe Not Found 😔\n\n"
                           "The recipe you're looking for is not in our database.\n\n"
                           "### 💡 Suggestions:\n"
                           "- Try different keywords\n"
                           "- Check the recipe spelling\n"
                           "- Search for similar recipes\n\n"
                           "**Would you like me to recommend other popular recipes?** 🍽️",
                    "score": 0.0,
                    "not_found": True
                }]
        
        # ==========================================
        # ERROR HANDLING: Low Similarity Threshold
        # ==========================================
        # Jika score tertinggi terlalu rendah (< 0.3), anggap tidak relevan
        if points[0].score < 0.3:
            language = detect_language(query)
            
            if language == "indonesian":
                return [{
                    "recipe_name": "Tidak Relevan",
                    "text": "## Hmm, Tidak Ada yang Cocok 🤔\n\n"
                           f"Saya menemukan beberapa resep, tapi tidak ada yang cocok dengan **'{query}'**.\n\n"
                           "### 💡 Coba:\n"
                           "- Gunakan nama resep yang lebih spesifik\n"
                           "- Cari berdasarkan bahan utama\n"
                           "- Tanyakan kategori makanan (misal: 'kue', 'ayam', 'pasta')\n\n"
                           "**Atau mau saya carikan resep populer?** 🍳",
                    "score": points[0].score,
                    "not_found": True
                }]
            else:
                return [{
                    "recipe_name": "Not Relevant",
                    "text": "## Hmm, No Match Found 🤔\n\n"
                           f"I found some recipes, but none match **'{query}'** well.\n\n"
                           "### 💡 Try:\n"
                           "- Use more specific recipe names\n"
                           "- Search by main ingredient\n"
                           "- Ask for food categories (e.g., 'cake', 'chicken', 'pasta')\n\n"
                           "**Or would you like popular recipe suggestions?** 🍳",
                    "score": points[0].score,
                    "not_found": True
                }]

        # ==========================================
        # Process Valid Results
        # ==========================================
        hits = []
        language = detect_language(query)

        for i, point in enumerate(points):
            payload = point.payload or {}
            recipe_name = payload.get("recipe_name", "Tanpa Judul")
            score = point.score
            
            # GUNAKAN FORMAT BARU YANG KONSISTEN
            formatted_text = format_recipe_output(payload, language)

            hits.append({
                "recipe_name": recipe_name,
                "text": formatted_text,
                "score": score,
                "not_found": False
            })

        return hits

    except Exception as e:
        print(f"🚨 ERROR SEARCH: {e}")
        language = detect_language(query)
        
        error_msg = {
            "recipe_name": "Error" if language == "english" else "Kesalahan",
            "text": f"## System Error ⚠️\n\n{str(e)}\n\nSilakan coba lagi atau hubungi admin." 
                   if language == "indonesian" 
                   else f"## System Error ⚠️\n\n{str(e)}\n\nPlease try again or contact admin.",
            "score": 0.0,
            "error": str(e)
        }
        return [error_msg]