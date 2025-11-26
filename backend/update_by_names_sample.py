#!/usr/bin/env python3
"""
Update a sample of recipes by name: for each recipe in the source JSON, find the point
in Qdrant by embedding the recipe name, and if payload lacks 'ingredients', run
rag_core.llm_extract_structured and set the payload.

Usage: PYTHONPATH=. python3 backend/update_by_names_sample.py --count 10
"""
import argparse
import time
import json
from backend import rag_core

def update_sample(count=10, sleep_between=0.3):
    client = rag_core.get_client()
    with open('backend/data/recipe_new.json','r',encoding='utf-8') as f:
        data = json.load(f)

    updated = 0
    processed = 0
    for item in data:
        if processed >= count:
            break
        if not isinstance(item, dict):
            continue
        name = item.get('name')
        if not name:
            continue
        processed += 1
        print(f"Processing sample [{processed}/{count}]: {name}")
        vec = rag_core.embed_text(name)
        res = client.query_points(collection_name=rag_core.COLLECTION_NAME, query=vec, limit=1, with_payload=True)
        if not res.points:
            print("  No point found for this name")
            continue
        pt = res.points[0]
        pid = getattr(pt, 'id', None)
        payload = pt.payload or {}
        if payload.get('ingredients'):
            print("  Already has ingredients, skipping")
            continue

        # extract using LLM helper
        extracted = rag_core.llm_extract_structured(payload.get('text',''), payload.get('recipe_name',name))
        new_ings = extracted.get('ingredients') or []
        new_instr = extracted.get('instructions') or []
        text_id = extracted.get('text_id') or None

        if new_ings:
            payload['ingredients'] = new_ings
        if new_instr:
            payload['instructions'] = new_instr
        if text_id:
            payload['text_id'] = text_id

        try:
            client.set_payload(collection_name=rag_core.COLLECTION_NAME, payload=payload, points=[pid])
            updated += 1
            print(f"  Updated point {pid}: +{len(new_ings)} ingredients")
        except Exception as e:
            print(f"  Failed to set payload for {pid}: {e}")

        time.sleep(sleep_between)

    print(f"Sample update done. Updated {updated} of {processed} processed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=10)
    args = parser.parse_args()
    update_sample(count=args.count)
