"""Scan Qdrant collection for noisy ingredient payloads and prepare fixes.

Usage:
  PYTHONPATH=. python3 backend/scan_and_prepare_updates.py [--limit N] [--apply]

Default: dry-run (no writes). Add `--apply` to perform updates (will attempt per-point update).
"""
from rag_core import get_client, COLLECTION_NAME, extract_ingredients_from_text, llm_extract_structured, NUTRITION_HINTS, TIME_TEMPERATURE_PATTERN
import re
import json
import argparse


def is_noisy_ingredient(ing: str) -> bool:
    if not ing or len(ing.strip()) < 2:
        return True
    # starts with common verb -> noisy
    if re.match(r'^(add|stir|mix|fold|press|place|roll|peel|slice|melt|pour|press|set|make|brush|bake|preheat)\b', ing.strip(), re.IGNORECASE):
        return True
    # contains nutrition tokens
    if NUTRITION_HINTS.search(ing) or TIME_TEMPERATURE_PATTERN.search(ing):
        return True
    # single short tokens like 'core', 'some', 'way' are likely noise
    if len(ing.split()) == 1 and len(re.sub(r'[^A-Za-z]','', ing)) <= 4:
        return True
    return False


def scan_collection(limit=None):
    client = get_client()
    offset = 0
    batch = 100
    found = []

    while True:
        resp = client.scroll(collection_name=COLLECTION_NAME, limit=batch, offset=offset, with_payload=True)
        points = None
        # support different qdrant client shapes
        if hasattr(resp, 'points'):
            points = resp.points
        elif isinstance(resp, dict) and 'result' in resp:
            points = resp['result']
        elif isinstance(resp, dict) and 'points' in resp:
            points = resp['points']
        else:
            points = resp

        if not points:
            break

        for p in points:
            # points may be returned as objects, dicts, or nested lists depending on client version
            items = [p]
            if isinstance(p, (list, tuple)):
                items = list(p)
            for item in items:
                pid = None
                payload = {}
                if hasattr(item, 'id'):
                    pid = item.id
                    payload = item.payload if hasattr(item, 'payload') else {}
                elif isinstance(item, dict):
                    pid = item.get('id')
                    payload = item.get('payload') or {}
                else:
                    # unknown shape, skip
                    continue
            ingredients = payload.get('ingredients')
            text = payload.get('text','')
            recipe_name = payload.get('recipe_name') or payload.get('title') or '<no-name>'

            # normalize ingredients list to python list of strings
            ings = []
            if ingredients:
                if isinstance(ingredients, list):
                    ings = [str(x).strip() for x in ingredients if str(x).strip()]
                else:
                    ings = [ln.strip() for ln in str(ingredients).split('\n') if ln.strip()]

            noisy = False
            # If there are ingredients but many are noisy, mark for update
            if ings:
                noisy_count = sum(1 for it in ings if is_noisy_ingredient(it))
                if noisy_count >= max(1, len(ings)//2):
                    noisy = True
            else:
                # if no ingredients, attempt extraction and update
                noisy = True

                if noisy:
                    found.append({'id': pid, 'recipe_name': recipe_name, 'payload_ings': ings, 'text': text})

            if limit and len(found) >= limit:
                return found

        # advance
        offset += batch

    return found


def prepare_updates(candidates, use_llm=False):
    prepared = []
    for c in candidates:
        text = c['text']
        name = c['recipe_name']
        if use_llm:
            try:
                res = llm_extract_structured(text, name, target_language='indonesian')
                new_ings = res.get('ingredients', [])
                new_instr = res.get('instructions', [])
            except Exception:
                new_ings = extract_ingredients_from_text(text)
                new_instr = []
        else:
            new_ings = extract_ingredients_from_text(text)
            new_instr = []

        prepared.append({'id': c['id'], 'recipe_name': name, 'new_ingredients': new_ings, 'new_instructions': new_instr})
    return prepared


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=200, help='Max candidates to inspect')
    parser.add_argument('--apply', action='store_true', help='Apply updates to Qdrant')
    parser.add_argument('--use-llm', action='store_true', help='Use LLM for extraction when available')
    args = parser.parse_args()

    print('Scanning collection for noisy ingredient payloads...')
    candidates = scan_collection(limit=args.limit)
    print(f'Found {len(candidates)} candidate(s) that look noisy or missing ingredients.')

    if not candidates:
        return

    prepared = prepare_updates(candidates, use_llm=args.use_llm)

    # show summary
    for p in prepared[:50]:
        print('\n---')
        print(f"ID: {p['id']} | {p['recipe_name']}")
        print('New Ingredients:')
        for ni in p['new_ingredients'][:20]:
            print(' -', ni)
        if not p['new_ingredients']:
            print(' - (none found)')

    # write to file for review
    with open('backend/scan_update_preview.json', 'w') as f:
        json.dump(prepared, f, ensure_ascii=False, indent=2)

    print('\nPreview saved to backend/scan_update_preview.json')

    if args.apply:
        client = get_client()
        updated = 0
        for p in prepared:
            try:
                payload = {}
                if p['new_ingredients']:
                    payload['ingredients'] = p['new_ingredients']
                if p['new_instructions']:
                    payload['instructions'] = p['new_instructions']
                if not payload:
                    continue
                # Attempt to update point payload
                try:
                    client.update_point(collection_name=COLLECTION_NAME, point_id=p['id'], payload=payload)
                    updated += 1
                except Exception:
                    # fallback: upsert with empty vector (safe only if collection allows)
                    client.upsert(collection_name=COLLECTION_NAME, points=[{'id': p['id'], 'payload': payload}])
                    updated += 1
            except Exception as e:
                print('Failed to update', p['id'], e)

        print(f'Applied updates: {updated}')


if __name__ == '__main__':
    main()
