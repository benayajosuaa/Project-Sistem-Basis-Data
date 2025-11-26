"""
RAG Evaluation Script
=====================
Mengevaluasi performa sistem RAG untuk recipe search.

Metrics yang diukur:
1. Retrieval Quality (Score distribution, accuracy)
2. Extraction Quality (Ingredients accuracy)
3. Response Time (Latency)
4. Overall Relevance (Manual + Automated)

Usage:
    python backend/evaluate_rag.py
    python backend/evaluate_rag.py --save-results
    python backend/evaluate_rag.py --test-case "Apple Pie"
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import sys

# Import RAG core functions
try:
    from rag_core import search_recipes, extract_ingredients_from_text
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from rag_core import search_recipes, extract_ingredients_from_text

# ==========================================
# TEST DATASET
# ==========================================
TEST_QUERIES = [
    {
        "query": "Apple Pie by Grandma Ople",
        "expected_recipe": "Apple Pie by Grandma Ople",
        "expected_ingredients": [
            "sugar", "flour", "butter", "apple", "cinnamon", "pastry"
        ],
        "language": "english"
    },
    {
        "query": "No-Bake Chocolate Coconut Cookies",
        "expected_recipe": "No-Bake Chocolate Coconut Cookies",
        "expected_ingredients": [
            "sugar", "butter", "milk", "cocoa", "coconut", "oat"
        ],
        "language": "english"
    },
    {
        "query": "Old Fashioned Cocktail",
        "expected_recipe": "Old Fashioned",
        "expected_ingredients": [
            "whiskey", "sugar", "bitters", "orange"
        ],
        "language": "english"
    },
    {
        "query": "Resep kue coklat",
        "expected_recipe": None,  # Generic query
        "expected_ingredients": ["coklat", "chocolate", "cocoa"],
        "language": "indonesian"
    },
    {
        "query": "Cara membuat pizza",
        "expected_recipe": None,
        "expected_ingredients": ["cheese", "dough", "tomato", "flour"],
        "language": "indonesian"
    },
    {
        "query": "chicken pasta recipe",
        "expected_recipe": None,
        "expected_ingredients": ["chicken", "pasta", "cheese"],
        "language": "english"
    },
    {
        "query": "cookies tanpa telur",
        "expected_recipe": None,
        "expected_ingredients": ["flour", "sugar", "butter"],
        "language": "indonesian"
    },
    {
        "query": "chocolate cake",
        "expected_recipe": None,
        "expected_ingredients": ["chocolate", "flour", "sugar", "egg"],
        "language": "english"
    }
]

# ==========================================
# EVALUATION METRICS
# ==========================================

def evaluate_retrieval_score(score: float) -> str:
    """Classify retrieval score quality"""
    if score >= 0.7:
        return "Excellent"
    elif score >= 0.5:
        return "Good"
    elif score >= 0.3:
        return "Fair"
    else:
        return "Poor"

def calculate_ingredient_overlap(extracted: List[str], expected: List[str]) -> float:
    """Calculate overlap between extracted and expected ingredients"""
    if not expected:
        return 1.0  # No ground truth to compare
    
    # Normalize to lowercase for comparison
    extracted_lower = set([ing.lower() for ing in extracted])
    expected_lower = set([exp.lower() for exp in expected])
    
    # Calculate overlap (how many expected ingredients were found)
    matches = 0
    for exp in expected_lower:
        # Check if any extracted ingredient contains the expected keyword
        if any(exp in ext for ext in extracted_lower):
            matches += 1
    
    # Precision: correct ingredients / total extracted
    precision = matches / len(extracted_lower) if extracted_lower else 0
    
    # Recall: correct ingredients / total expected
    recall = matches / len(expected_lower) if expected_lower else 0
    
    # F1 Score
    if precision + recall == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1

def evaluate_recipe_match(retrieved_name: str, expected_name: str) -> bool:
    """Check if retrieved recipe matches expected recipe"""
    if not expected_name:
        return True  # Generic query, no specific match expected
    
    # Normalize names
    retrieved_lower = retrieved_name.lower().strip()
    expected_lower = expected_name.lower().strip()
    
    # Check if they match (fuzzy match)
    return expected_lower in retrieved_lower or retrieved_lower in expected_lower

# ==========================================
# MAIN EVALUATION FUNCTION
# ==========================================

def evaluate_rag(test_queries: List[Dict] = None, verbose: bool = True) -> Dict[str, Any]:
    """
    Run comprehensive RAG evaluation
    
    Returns:
        Dict containing all evaluation metrics
    """
    if test_queries is None:
        test_queries = TEST_QUERIES
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "total_queries": len(test_queries),
        "test_cases": [],
        "metrics": {
            "avg_retrieval_score": 0.0,
            "avg_ingredient_f1": 0.0,
            "avg_response_time": 0.0,
            "recipe_match_rate": 0.0,
            "score_distribution": {
                "excellent": 0,
                "good": 0,
                "fair": 0,
                "poor": 0
            }
        }
    }
    
    total_score = 0.0
    total_f1 = 0.0
    total_time = 0.0
    recipe_matches = 0
    
    print("\n" + "="*70)
    print("🔍 RAG EVALUATION STARTED")
    print("="*70)
    
    for i, test in enumerate(test_queries, 1):
        if verbose:
            print(f"\n[{i}/{len(test_queries)}] Testing: '{test['query']}'")
        
        # Measure response time
        start_time = time.time()
        try:
            search_results = search_recipes(test['query'], top_k=3)
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
        elapsed = time.time() - start_time
        
        if not search_results or search_results[0].get("not_found") or search_results[0].get("error"):
            if verbose:
                print(f"  ⚠️  No valid results found")
            results["test_cases"].append({
                "query": test['query'],
                "status": "no_results",
                "response_time": elapsed
            })
            continue
        
        # Get top result
        top_result = search_results[0]
        score = top_result.get('score', 0.0)
        recipe_name = top_result.get('recipe_name', '')
        ingredients = top_result.get('ingredients', [])
        
        # Evaluate retrieval score
        score_quality = evaluate_retrieval_score(score)
        results["metrics"]["score_distribution"][score_quality.lower()] += 1
        
        # Evaluate recipe match
        is_match = evaluate_recipe_match(recipe_name, test.get('expected_recipe'))
        if is_match:
            recipe_matches += 1
        
        # Evaluate ingredient extraction
        f1_score = calculate_ingredient_overlap(ingredients, test['expected_ingredients'])
        
        # Accumulate metrics
        total_score += score
        total_f1 += f1_score
        total_time += elapsed
        
        # Store test case result
        test_result = {
            "query": test['query'],
            "expected_recipe": test.get('expected_recipe'),
            "retrieved_recipe": recipe_name,
            "recipe_match": is_match,
            "retrieval_score": round(score, 4),
            "score_quality": score_quality,
            "ingredient_f1": round(f1_score, 4),
            "response_time": round(elapsed, 4),
            "extracted_ingredients": ingredients[:5],  # Show first 5
            "status": "success"
        }
        results["test_cases"].append(test_result)
        
        # Verbose output
        if verbose:
            print(f"  📊 Score: {score:.4f} ({score_quality})")
            print(f"  🎯 Recipe Match: {'✅' if is_match else '❌'} {recipe_name}")
            print(f"  🧪 Ingredient F1: {f1_score:.4f}")
            print(f"  ⏱️  Response Time: {elapsed:.4f}s")
            if ingredients:
                print(f"  📝 Ingredients: {', '.join(ingredients[:3])}{'...' if len(ingredients) > 3 else ''}")
    
    # Calculate averages
    valid_tests = len([tc for tc in results["test_cases"] if tc["status"] == "success"])
    if valid_tests > 0:
        results["metrics"]["avg_retrieval_score"] = round(total_score / valid_tests, 4)
        results["metrics"]["avg_ingredient_f1"] = round(total_f1 / valid_tests, 4)
        results["metrics"]["avg_response_time"] = round(total_time / valid_tests, 4)
        results["metrics"]["recipe_match_rate"] = round(recipe_matches / valid_tests, 4)
    
    # Print summary
    print("\n" + "="*70)
    print("📈 EVALUATION SUMMARY")
    print("="*70)
    print(f"✅ Valid Tests: {valid_tests}/{len(test_queries)}")
    print(f"📊 Avg Retrieval Score: {results['metrics']['avg_retrieval_score']:.4f}")
    print(f"🎯 Recipe Match Rate: {results['metrics']['recipe_match_rate']*100:.1f}%")
    print(f"🧪 Avg Ingredient F1: {results['metrics']['avg_ingredient_f1']:.4f}")
    print(f"⏱️  Avg Response Time: {results['metrics']['avg_response_time']:.4f}s")
    print(f"\n🏆 Score Distribution:")
    for quality, count in results["metrics"]["score_distribution"].items():
        print(f"  {quality.capitalize()}: {count}")
    print("="*70 + "\n")
    
    return results

# ==========================================
# SCORE THRESHOLD ANALYZER
# ==========================================

def analyze_score_threshold(test_queries: List[Dict] = None) -> None:
    """Analyze different score thresholds to find optimal value"""
    if test_queries is None:
        test_queries = TEST_QUERIES
    
    print("\n" + "="*70)
    print("🔬 SCORE THRESHOLD ANALYSIS")
    print("="*70)
    
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]
    
    for threshold in thresholds:
        accepted = 0
        rejected = 0
        
        for test in test_queries:
            try:
                results = search_recipes(test['query'], top_k=1)
                if results and not results[0].get("not_found"):
                    score = results[0].get('score', 0.0)
                    if score >= threshold:
                        accepted += 1
                    else:
                        rejected += 1
            except Exception:
                continue
        
        total = accepted + rejected
        if total > 0:
            acceptance_rate = (accepted / total) * 100
            print(f"Threshold {threshold:.1f}: {accepted}/{total} accepted ({acceptance_rate:.1f}%)")
    
    print("\n💡 Current threshold: 0.3 (set in rag_core.py line 652)")
    print("="*70 + "\n")

# ==========================================
# SAVE RESULTS
# ==========================================

def save_evaluation_results(results: Dict, output_path: str = None) -> None:
    """Save evaluation results to JSON file"""
    if output_path is None:
        output_path = Path(__file__).parent / "evaluation_results.json"
    else:
        output_path = Path(output_path)
    
    # Load existing results if file exists
    if output_path.exists():
        with open(output_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
            if not isinstance(history, list):
                history = [history]
    else:
        history = []
    
    # Append new results
    history.append(results)
    
    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Results saved to: {output_path}")

# ==========================================
# CLI
# ==========================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate RAG system performance")
    parser.add_argument("--save-results", action="store_true", help="Save results to JSON file")
    parser.add_argument("--analyze-threshold", action="store_true", help="Analyze score thresholds")
    parser.add_argument("--test-case", type=str, help="Test single query")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    
    args = parser.parse_args()
    
    if args.test_case:
        # Single test case
        test = {
            "query": args.test_case,
            "expected_recipe": None,
            "expected_ingredients": [],
            "language": "english"
        }
        results = evaluate_rag([test], verbose=not args.quiet)
    else:
        # Full evaluation
        results = evaluate_rag(verbose=not args.quiet)
    
    if args.analyze_threshold:
        analyze_score_threshold()
    
    if args.save_results:
        save_evaluation_results(results)
