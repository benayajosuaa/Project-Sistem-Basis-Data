import json
from rag_core import model, client, COLLECTION_NAME

# === 1. Load Dataset Evaluasi ===
def load_eval_dataset(path="eval_dataset.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# === 2. Precision@k ===
def precision_at_k(retrieved, ground_truth_id, k=3):
    return 1 if ground_truth_id in retrieved[:k] else 0

# === 3. Evaluate ===
def evaluate_vector_db(k=3):
    dataset = load_eval_dataset()
    scores = []

    print("\n=== Starting Vector DB Evaluation ===\n")

    for item in dataset:
        query = item["query"]
        expected_id = item["ground_truth_id"]  # ID dokumen yg benar di Qdrant

        # Embedding query
        q_vec = model.encode([query])[0].tolist()

        # Search ke Qdrant
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=q_vec,
            limit=k
        )

        retrieved_ids = [r.id for r in results]

        # Hitung precision
        p = precision_at_k(retrieved_ids, expected_id, k)
        scores.append(p)

        print(f"Query: {query}")
        print(f"Ground truth ID: {expected_id}")
        print(f"Retrieved: {retrieved_ids}")
        print(f"Precision@{k}: {p}\n")

    print("=== Final Evaluation Result ===")
    print(f"Average Precision@{k}: {sum(scores)/len(scores):.3f}")

if __name__ == "__main__":
    evaluate_vector_db()
