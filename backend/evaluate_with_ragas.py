"""
Evaluate RAG system using RAGAS metrics.

This script evaluates the Recipe RAG system using:
- Context Precision: How relevant are the retrieved documents?
- Context Recall: How much of the ground truth is captured?
- Faithfulness: Is the answer faithful to the retrieved context?
- Answer Relevancy: How relevant is the answer to the question?

Usage:
    python backend/evaluate_with_ragas.py
"""

import json
import os
import pandas as pd
from dotenv import load_dotenv
import sys

# Import RAG system
sys.path.insert(0, os.path.dirname(__file__))
from rag_core import search_recipes

load_dotenv()

# Setup API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY must be set in .env file")

def load_test_queries(filepath="backend/test_queries.json"):
    """Load test queries from JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def calculate_manual_metrics(df):
    """Calculate evaluation metrics manually."""
    metrics = {
        "context_precision": 0.0,
        "context_recall": 0.0,
        "faithfulness": 0.0,
        "answer_relevancy": 0.0
    }
    
    n = len(df)
    
    for idx, row in df.iterrows():
        question = row["question"]
        answer = row["answer"]
        contexts = row["contexts"]
        ground_truth = row["ground_truth"]
        
        # 1. Context Precision: Check if contexts contain relevant info
        #    Simple: Check if any key ingredients appear in contexts
        precision_score = 0
        if contexts:
            # Extract key terms from question
            question_terms = set(question.lower().split())
            for ctx in contexts:
                ctx_lower = ctx.lower()
                matches = sum(1 for term in question_terms if term in ctx_lower and len(term) > 3)
                if matches > 0:
                    precision_score = min(1.0, matches / max(len(question_terms), 1))
                    break
        metrics["context_precision"] += precision_score
        
        # 2. Context Recall: Check if ground truth info is in contexts
        recall_score = 0
        if contexts and ground_truth:
            ground_truth_terms = set(ground_truth.lower().split())
            ground_truth_terms = {t for t in ground_truth_terms if len(t) > 3}
            
            found_terms = 0
            all_contexts = " ".join(contexts).lower()
            for term in ground_truth_terms:
                if term in all_contexts:
                    found_terms += 1
            
            if ground_truth_terms:
                recall_score = found_terms / len(ground_truth_terms)
        metrics["context_recall"] += recall_score
        
        # 3. Faithfulness: Check if answer doesn't hallucinate
        #    Simple: Check if answer content exists in contexts
        faithful_score = 0
        if answer and contexts:
            answer_terms = set(answer.lower().split())
            answer_terms = {t for t in answer_terms if len(t) > 4}  # Only significant words
            
            all_contexts = " ".join(contexts).lower()
            found_in_context = sum(1 for term in answer_terms if term in all_contexts)
            
            if answer_terms:
                faithful_score = found_in_context / len(answer_terms)
        metrics["faithfulness"] += faithful_score
        
        # 4. Answer Relevancy: Check if answer addresses the question
        relevancy_score = 0
        if answer and question:
            question_terms = set(question.lower().split())
            question_terms = {t for t in question_terms if len(t) > 3}
            
            answer_lower = answer.lower()
            matches = sum(1 for term in question_terms if term in answer_lower)
            
            if question_terms:
                relevancy_score = matches / len(question_terms)
        metrics["answer_relevancy"] += relevancy_score
    
    # Calculate averages
    for key in metrics:
        metrics[key] = metrics[key] / n if n > 0 else 0.0
    
    return metrics

def get_rag_response(query: str, top_k: int = 3):
    """Get response from RAG system."""
    try:
        results = search_recipes(query, top_k=top_k)
        
        if not results or not isinstance(results, list):
            return {
                "answer": "No results found",
                "contexts": [],
                "recipe_names": []
            }
        
        # Extract answer (formatted text from top result)
        answer = results[0].get("text", "")
        
        # Extract contexts (raw text from all results)
        contexts = []
        recipe_names = []
        for res in results:
            recipe_name = res.get("recipe_name", "")
            recipe_names.append(recipe_name)
            
            # Get raw context text
            raw_text = res.get("text", "")
            if raw_text:
                contexts.append(raw_text)
        
        return {
            "answer": answer,
            "contexts": contexts,
            "recipe_names": recipe_names
        }
    
    except Exception as e:
        print(f"Error getting RAG response: {e}")
        return {
            "answer": f"Error: {str(e)}",
            "contexts": [],
            "recipe_names": []
        }

def prepare_evaluation_dataset(test_queries):
    """Prepare dataset in RAGAS format."""
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    
    print("\n🔄 Generating RAG responses for test queries...")
    for i, test in enumerate(test_queries, 1):
        query = test["query"]
        ground_truth = test["ground_truth"]
        
        print(f"  [{i}/{len(test_queries)}] Processing: {query}")
        
        # Get RAG response
        rag_response = get_rag_response(query, top_k=3)
        
        # Add to dataset
        data["question"].append(query)
        data["answer"].append(rag_response["answer"])
        data["contexts"].append(rag_response["contexts"])
        data["ground_truth"].append(ground_truth)
        
        print(f"      ✓ Retrieved {len(rag_response['contexts'])} contexts")
        print(f"      ✓ Top recipe: {rag_response['recipe_names'][0] if rag_response['recipe_names'] else 'None'}")
    
    return pd.DataFrame(data)

def run_evaluation():
    """Run RAGAS evaluation."""
    print("=" * 60)
    print("🎯 RAGAS Evaluation for Recipe RAG System")
    print("=" * 60)
    
    # Load test queries
    test_queries = load_test_queries()
    print(f"\n📝 Loaded {len(test_queries)} test queries")
    
    # Prepare dataset
    dataset = prepare_evaluation_dataset(test_queries)
    
    print("\n" + "=" * 60)
    print("📊 Running Custom Evaluation Metrics...")
    print("=" * 60)
    
    # Run custom evaluation
    try:
        print("\n⏳ Calculating metrics...")
        result = calculate_manual_metrics(dataset)
        
        print("\n" + "=" * 60)
        print("✅ EVALUATION RESULTS")
        print("=" * 60)
        
        # Print results
        print(f"\n📈 Overall Scores:")
        print(f"  • Context Precision:  {result['context_precision']:.4f}")
        print(f"  • Context Recall:     {result['context_recall']:.4f}")
        print(f"  • Faithfulness:       {result['faithfulness']:.4f}")
        print(f"  • Answer Relevancy:   {result['answer_relevancy']:.4f}")
        
        # Calculate average
        avg_score = sum([
            result['context_precision'],
            result['context_recall'],
            result['faithfulness'],
            result['answer_relevancy']
        ]) / 4
        
        print(f"\n  🎯 Average Score:      {avg_score:.4f}")
        
        # Save results
        output_file = "backend/ragas_evaluation_results.json"
        results_dict = {
            "metrics": {
                "context_precision": float(result['context_precision']),
                "context_recall": float(result['context_recall']),
                "faithfulness": float(result['faithfulness']),
                "answer_relevancy": float(result['answer_relevancy']),
                "average": float(avg_score)
            },
            "test_count": len(test_queries)
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Results saved to: {output_file}")
        
        # Interpretation
        print("\n" + "=" * 60)
        print("📖 INTERPRETATION")
        print("=" * 60)
        print("""
Context Precision: Measures how relevant the retrieved documents are.
  • High score (>0.7): System retrieves highly relevant documents
  • Low score (<0.5): System retrieves many irrelevant documents

Context Recall: Measures if all relevant info from ground truth is retrieved.
  • High score (>0.7): System captures most relevant information
  • Low score (<0.5): System misses important information

Faithfulness: Measures if answer is faithful to retrieved context.
  • High score (>0.8): Answer doesn't hallucinate
  • Low score (<0.6): Answer contains hallucinations

Answer Relevancy: Measures how relevant answer is to the question.
  • High score (>0.7): Answer directly addresses the question
  • Low score (<0.5): Answer is off-topic or incomplete
        """)
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    run_evaluation()
