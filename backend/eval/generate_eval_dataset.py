import json
import random
from rag_core import client, COLLECTION_NAME, model

OUTPUT_FILE = "eval_dataset.json"

def generate_eval_dataset(sample_size=10):
    # Ambil semua ID dokumen
    points = client.scroll(collection_name=COLLECTION_NAME, limit=99999)[0]

    print(f"Total dokumen: {len(points)}")

    # Pilih sample acak
    samples = random.sample(points, sample_size)

    eval_data = []

    for p in samples:
        text = p.payload["text"]

        # Query otomatis → ambil 8 kata pertama
        query = " ".join(text.split()[:8])

        eval_data.append({
            "query": query,
            "ground_truth_id": p.id
        })

    # Save dataset
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=4, ensure_ascii=False)

    print(f"Selesai membuat {OUTPUT_FILE} dengan {len(eval_data)} item.")

if __name__ == "__main__":
    generate_eval_dataset()
