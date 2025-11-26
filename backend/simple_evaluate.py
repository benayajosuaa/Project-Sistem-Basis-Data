"""
Simple RAG Evaluation using HTTP API and custom metrics.

This evaluates the Recipe RAG system by:
1. Sending queries via HTTP API
2. Calculating custom retrieval & generation metrics
3. Generating evaluation report

Usage:
    python backend/simple_evaluate.py
"""

import json
import requests
import pandas as pd
from collections import Counter

API_URL = "http://localhost:8000/ask_raw"

def load_test_queries(filepath="backend/test_queries.json"):
    """Load test queries from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def get_rag_response_via_api(query: str, top_k: int = 3):
    """Get response from RAG system via HTTP API."""
    try:
        response = requests.post(
            API_URL,
            json={"question": query, "top_k": top_k},
            timeout=30
        )
        response.raise_for_status()
        results = response.json()
        
        if not results or not isinstance(results, list):
            return {
                "answer": "No results found",
                "contexts": [],
                "recipe_names": [],
                "scores": []
            }
        
        # Extract data from results
        answer = results[0].get("text", "")
        contexts = [res.get("text", "") for res in results]
        recipe_names = [res.get("recipe_name", "") for res in results]
        scores = [res.get("score", 0.0) for res in results]
        
        return {
            "answer": answer,
            "contexts": contexts,
            "recipe_names": recipe_names,
            "scores": scores
        }
    
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return {
            "answer": f"Error: {str(e)}",
            "contexts": [],
            "recipe_names": [],
            "scores": []
        }

def calculate_retrieval_metrics(test_queries, responses):
    """Calculate retrieval-focused metrics."""
    print("\n" + "=" * 60)
    print("📊 RETRIEVAL METRICS")
    print("=" * 60)
    
    total_queries = len(test_queries)
    correct_retrievals = 0
    avg_top1_score = 0
    avg_top3_score = 0
    mrr_sum = 0  # Mean Reciprocal Rank
    
    for i, (test, response) in enumerate(zip(test_queries, responses)):
        expected_recipe = test.get("expected_recipe", "")
        retrieved_names = response.get("recipe_names", [])
        scores = response.get("scores", [])
        
        # Check if expected recipe is in top-k
        if expected_recipe:
            try:
                rank = next((idx + 1 for idx, name in enumerate(retrieved_names) 
                           if expected_recipe.lower() in name.lower()), 0)
                if rank > 0:
                    correct_retrievals += 1
                    mrr_sum += 1.0 / rank  # MRR calculation
            except:
                pass
        
        # Average scores
        if scores:
            avg_top1_score += scores[0]
            avg_top3_score += sum(scores) / len(scores)
    
    # Calculate final metrics
    precision_at_1 = correct_retrievals / total_queries if total_queries > 0 else 0
    mrr = mrr_sum / total_queries if total_queries > 0 else 0
    avg_top1_score = avg_top1_score / total_queries if total_queries > 0 else 0
    avg_top3_score = avg_top3_score / total_queries if total_queries > 0 else 0
    
    print(f"\n📈 Results:")
    print(f"  • Precision@1 (Correct Top Result):  {precision_at_1:.4f} ({correct_retrievals}/{total_queries})")
    print(f"  • Mean Reciprocal Rank (MRR):        {mrr:.4f}")
    print(f"  • Avg Similarity Score (Top-1):      {avg_top1_score:.4f}")
    print(f"  • Avg Similarity Score (Top-3):      {avg_top3_score:.4f}")
    
    return {
        "precision_at_1": precision_at_1,
        "mrr": mrr,
        "avg_similarity_top1": avg_top1_score,
        "avg_similarity_top3": avg_top3_score
    }

def calculate_generation_metrics(test_queries, responses):
    """Calculate generation-focused metrics."""
    print("\n" + "=" * 60)
    print("📝 GENERATION METRICS")
    print("=" * 60)
    
    total = len(test_queries)
    context_recall_sum = 0
    faithfulness_sum = 0
    answer_relevancy_sum = 0
    completeness_sum = 0
    
    for test, response in zip(test_queries, responses):
        question = test["query"]
        ground_truth = test["ground_truth"]
        key_ingredients = test.get("key_ingredients", [])
        answer = response.get("answer", "")
        contexts = response.get("contexts", [])
        
        # 1. Context Recall: How much ground truth is captured in contexts?
        if ground_truth and contexts:
            gt_terms = set(ground_truth.lower().split())
            gt_terms = {t for t in gt_terms if len(t) > 3}
            
            all_contexts = " ".join(contexts).lower()
            found = sum(1 for term in gt_terms if term in all_contexts)
            context_recall = found / len(gt_terms) if gt_terms else 0
            context_recall_sum += context_recall
        
        # 2. Faithfulness: Answer doesn't hallucinate (content in context)
        if answer and contexts:
            answer_terms = set(answer.lower().split())
            answer_terms = {t for t in answer_terms if len(t) > 4}
            
            all_contexts = " ".join(contexts).lower()
            found = sum(1 for term in answer_terms if term in all_contexts)
            faithfulness = found / len(answer_terms) if answer_terms else 0
            faithfulness_sum += faithfulness
        
        # 3. Answer Relevancy: Answer addresses the question
        if answer and question:
            q_terms = set(question.lower().split())
            q_terms = {t for t in q_terms if len(t) > 3}
            
            answer_lower = answer.lower()
            matches = sum(1 for term in q_terms if term in answer_lower)
            relevancy = matches / len(q_terms) if q_terms else 0
            answer_relevancy_sum += relevancy
        
        # 4. Completeness: Check if key ingredients mentioned
        if key_ingredients and answer:
            answer_lower = answer.lower()
            found_ing = sum(1 for ing in key_ingredients if ing.lower() in answer_lower)
            completeness = found_ing / len(key_ingredients) if key_ingredients else 0
            completeness_sum += completeness
    
    # Calculate averages
    context_recall = context_recall_sum / total if total > 0 else 0
    faithfulness = faithfulness_sum / total if total > 0 else 0
    answer_relevancy = answer_relevancy_sum / total if total > 0 else 0
    completeness = completeness_sum / total if total > 0 else 0
    
    print(f"\n📈 Results:")
    print(f"  • Context Recall:       {context_recall:.4f}")
    print(f"  • Faithfulness:         {faithfulness:.4f}")
    print(f"  • Answer Relevancy:     {answer_relevancy:.4f}")
    print(f"  • Completeness:         {completeness:.4f}")
    
    return {
        "context_recall": context_recall,
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "completeness": completeness
    }

def run_evaluation():
    """Run complete evaluation."""
    print("=" * 60)
    print("🎯 RAG SYSTEM EVALUATION")
    print("=" * 60)
    
    # Load test queries
    test_queries = load_test_queries()
    print(f"\n📝 Loaded {len(test_queries)} test queries")
    
    # Get responses
    print("\n🔄 Getting RAG responses...")
    responses = []
    for i, test in enumerate(test_queries, 1):
        query = test["query"]
        print(f"  [{i}/{len(test_queries)}] {query}")
        response = get_rag_response_via_api(query, top_k=3)
        responses.append(response)
        if response["recipe_names"]:
            print(f"      ✓ Top result: {response['recipe_names'][0]} (score: {response['scores'][0]:.4f})")
        else:
            print(f"      ⚠️  No results")
    
    # Calculate metrics
    retrieval_metrics = calculate_retrieval_metrics(test_queries, responses)
    generation_metrics = calculate_generation_metrics(test_queries, responses)
    
    # Combined results
    print("\n" + "=" * 60)
    print("🎯 OVERALL SCORES")
    print("=" * 60)
    
    all_metrics = {**retrieval_metrics, **generation_metrics}
    
    # Calculate overall score
    key_metrics = [
        retrieval_metrics["precision_at_1"],
        retrieval_metrics["mrr"],
        generation_metrics["context_recall"],
        generation_metrics["faithfulness"],
        generation_metrics["answer_relevancy"]
    ]
    overall_score = sum(key_metrics) / len(key_metrics)
    
    print(f"\n  🏆 Overall RAG Score: {overall_score:.4f}")
    
    # Save results
    output_file = "backend/evaluation_results.json"
    results = {
        "overall_score": overall_score,
        "retrieval_metrics": retrieval_metrics,
        "generation_metrics": generation_metrics,
        "test_count": len(test_queries)
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Interpretation
    print("\n" + "=" * 60)
    print("📖 INTERPRETATION")
    print("=" * 60)
    print(f"""
RETRIEVAL METRICS:
  • Precision@1: {retrieval_metrics['precision_at_1']:.2%} - Correct recipe in top result
  • MRR: {retrieval_metrics['mrr']:.4f} - Average position of correct answer
  • Similarity: {retrieval_metrics['avg_similarity_top1']:.4f} - Vector embedding match quality

GENERATION METRICS:
  • Context Recall: {generation_metrics['context_recall']:.2%} - Coverage of ground truth
  • Faithfulness: {generation_metrics['faithfulness']:.2%} - No hallucinations
  • Answer Relevancy: {generation_metrics['answer_relevancy']:.2%} - Addresses question
  • Completeness: {generation_metrics['completeness']:.2%} - Key ingredients included

OVERALL: {overall_score:.2%} - {'✅ GOOD' if overall_score >= 0.7 else '⚠️ NEEDS IMPROVEMENT' if overall_score >= 0.5 else '❌ POOR'}
    """)
    
    return results

if __name__ == "__main__":
    try:
        run_evaluation()
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
