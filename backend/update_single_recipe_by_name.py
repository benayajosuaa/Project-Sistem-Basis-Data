"""Find a recipe by name in Qdrant, extract cleaned ingredients/instructions (LLM preferred), and update the point payload.

Usage:
  PYTHONPATH=. python3 backend/update_single_recipe_by_name.py --name "Old Fashioned Cocktail"
Optional flags: --use-llm to prefer LLM extraction
"""
import argparse
from rag_core import get_client, COLLECTION_NAME, llm_extract_structured, extract_ingredients_from_text, extract_steps


def find_and_update(name: str, use_llm: bool = True):
    client = get_client()
    offset = 0
    batch = 100
    found_any = False

    while True:
        resp = client.scroll(collection_name=COLLECTION_NAME, limit=batch, offset=offset, with_payload=True)
        points = None
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
            # support object or dict shapes
            item = p
            if isinstance(p, (list, tuple)):
                item = p[0]

            payload = item.payload if hasattr(item, 'payload') else (item.get('payload') if isinstance(item, dict) else {})
            recipe_name = payload.get('recipe_name') or payload.get('name') or payload.get('title')
            if recipe_name and recipe_name.strip().lower() == name.strip().lower():
                found_any = True
                pid = item.id if hasattr(item, 'id') else item.get('id')
                raw_text = payload.get('text', '')
                print(f'Found point id={pid} name={recipe_name}')

                if use_llm:
                    try:
                        res = llm_extract_structured(raw_text, recipe_name, target_language='indonesian')
                        new_ings = res.get('ingredients', [])
                        new_instr = res.get('instructions', [])
                    except Exception as e:
                        print('LLM failed, falling back to heuristics:', e)
                        new_ings = extract_ingredients_from_text(raw_text)
                        new_instr = extract_steps(raw_text, recipe_name)
                else:
                    new_ings = extract_ingredients_from_text(raw_text)
                    new_instr = extract_steps(raw_text, recipe_name)

                payload_update = {}
                if new_ings:
                    payload_update['ingredients'] = new_ings
                if new_instr:
                    payload_update['instructions'] = new_instr

                if not payload_update:
                    print('No structured data found to write.')
                else:
                    try:
                        client.update_point(collection_name=COLLECTION_NAME, point_id=pid, payload=payload_update)
                        print('Updated payload for', pid)
                    except Exception as e:
                        print('update_point failed, attempting upsert fallback:', e)
                        try:
                            client.upsert(collection_name=COLLECTION_NAME, points=[{'id': pid, 'payload': payload_update}])
                            print('Upsert fallback wrote payload for', pid)
                        except Exception as ee:
                            print('Upsert failed:', ee)

        offset += batch

    if not found_any:
        print('No recipe with that name found in collection.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True, help='Recipe name to find')
    parser.add_argument('--no-llm', action='store_true', help='Do not use LLM, use local heuristics')
    args = parser.parse_args()

    find_and_update(args.name, use_llm=not args.no_llm)


if __name__ == '__main__':
    main()
