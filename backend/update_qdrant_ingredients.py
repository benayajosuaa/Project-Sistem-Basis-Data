#!/usr/bin/env python3
"""
Batch re-extract and normalize ingredients/instructions for points in Qdrant.

Usage: python3 backend/update_qdrant_ingredients.py [--dry-run]

This script will iterate over points in the collection, and for points missing
`payload['ingredients']` or where the list looks too short/noisy, it will call
`rag_core.llm_extract_structured` (if available) and `rag_core.extract_ingredients_from_text` as fallback,
normalize results and upsert updated payloads back into Qdrant.
"""
import argparse
from backend import rag_core
from qdrant_client.models import PointStruct


def looks_noisy(ings: list) -> bool:
    if not ings:
        return True
    # if most items are single-char or numeric, consider noisy
    good = 0
    for i in ings:
        s = str(i).strip()
        if len(s) >= 3 and any(c.isalpha() for c in s):
            good += 1
    return (good / max(1, len(ings))) < 0.5


def main(dry_run=True, batch=256):
    client = rag_core.get_client()
    total_updated = 0
    print(f"Scanning collection: {rag_core.COLLECTION_NAME}")

    # Use scroll API to iterate; handle different shapes returned by client.scroll
    scroll_res = client.scroll(collection_name=rag_core.COLLECTION_NAME, with_payload=True, limit=500)
    items = []
    # scroll_res can be a list of dicts or a generator; normalize to flat list of point dicts
    try:
        for entry in scroll_res:
            if isinstance(entry, dict) and 'result' in entry:
                # some client versions wrap results
                items.extend(entry.get('result') or [])
            else:
                items.append(entry)
    except Exception:
        # fallback: try to convert to list directly
        try:
            items = list(scroll_res)
        except Exception:
            items = []

    updates = []

    for p in items:
        # p may be a dict with keys 'id','payload' or an object; handle both
        if isinstance(p, dict):
            payload = p.get('payload') or {}
            pid = p.get('id')
        else:
            # try attribute access
            payload = getattr(p, 'payload', None) or {}
            pid = getattr(p, 'id', None)
        # skip invalid entries
        if not pid:
            continue
        ings = payload.get('ingredients')

        need = False
        if not ings:
            need = True
        else:
            try:
                if looks_noisy(ings):
                    need = True
            except Exception:
                need = True

        if not need:
            continue

        print(f"Re-extracting for point {pid} (recipe={payload.get('recipe_name')})")
        try:
            res = rag_core.llm_extract_structured(payload.get('text',''), payload.get('recipe_name',''))
            new_ings = res.get('ingredients') or []
            new_instr = res.get('instructions') or []
            text_id = res.get('text_id') or None
            if new_ings:
                payload['ingredients'] = new_ings
            if new_instr:
                payload['instructions'] = new_instr
            if text_id:
                payload['text_id'] = text_id

            if not dry_run:
                # update payload in-place using set_payload (no vector required)
                try:
                    client.set_payload(collection_name=rag_core.COLLECTION_NAME, payload=payload, points=[pid])
                    total_updated += 1
                    if total_updated % batch == 0:
                        print(f"  Updated {total_updated} points so far")
                except Exception as e:
                    print(f"  Upsert/set_payload failed for {pid}: {e}")
            else:
                print(f"  [dry-run] would update: {len(new_ings)} ingredients, {len(new_instr)} instructions")
        except Exception as e:
            print(f"  Failed to extract: {e}")

    print(f"Done. Total updated: {total_updated} (dry_run={dry_run})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-dry-run', dest='dry', action='store_false', help='Actually upsert changes')
    parser.add_argument('--batch', type=int, default=256)
    args = parser.parse_args()
    main(dry_run=args.dry, batch=args.batch)
