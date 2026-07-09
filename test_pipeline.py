import os
import json
import pandas as pd
from rag_builder import FAISSVectorStore, build_vector_store_from_csv
from orchestrator import bi_workflow

def test_rag_retrieval():
    print("\n========================================")
    print("TESTING RAG INDEXING & RETRIEVAL")
    print("========================================")
    
    csv_file = "Sample - Superstore1.csv"
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return False
        
    print("Building FAISS vector database from CSV...")
    success = build_vector_store_from_csv(csv_file, "faiss_index")
    if not success:
        print("Failed to build FAISS vector database.")
        return False
        
    print("FAISS vector database built. Initializing search...")
    store = FAISSVectorStore("faiss_index")
    if not store.load_index():
        print("Failed to load FAISS index.")
        return False
        
    queries = [
        "Why are Tables unprofitable?",
        "What is the average discount for Furniture?",
        "Which product makes the biggest losses?"
    ]
    
    for q in queries:
        print(f"\nQuery: '{q}'")
        results = store.search(q, k=2)
        print(f"Retrieved {len(results)} facts:")
        for r in results:
            print(f"- {r}")
            
    return True

def test_orchestrator_pipeline():
    print("\n========================================")
    print("TESTING LANGGRAPH MULTI-AGENT PIPELINE")
    print("========================================")
    
    csv_file = "Sample - Superstore1.csv"
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return False
        
    initial_state = {
        "question": "Analyze my sales data, find why profit is down, predict next six months sales, and give recommendations.",
        "execution_plan": [],
        "history": [],
        "dataset_path": csv_file,
        "analysis_results": None,
        "root_cause_results": None,
        "forecast_results": None,
        "explanation_results": None,
        "recommendation_results": None,
        "final_report": None,
        "error": None
    }
    
    print("Invoking BI workflow...")
    result_state = bi_workflow.invoke(initial_state)
    
    if result_state.get("error"):
        print(f"Pipeline failed with error: {result_state['error']}")
        return False
        
    print("\nPipeline execution completed successfully!")
    print(f"Execution Plan: {result_state.get('execution_plan')}")
    
    analysis = result_state.get("analysis_results")
    if analysis:
        print(f"Analysis Agent Output:")
        print(f"- Total Sales: ${analysis.get('total_sales'):,.2f}")
        print(f"- Total Profit: ${analysis.get('total_profit'):,.2f}")
        print(f"- Profit Margin: {analysis.get('overall_margin'):.2f}%")
        
    root_cause = result_state.get("root_cause_results")
    if root_cause:
        print(f"Root Cause Agent Output:")
        print(f"- Loss Subcategories: {[x['Sub-Category'] for x in root_cause.get('loss_subcategories', [])]}")
        print(f"- Discount-Profit Correlation: {root_cause.get('correlation_discount_profit'):.4f}")
        
    forecast = result_state.get("forecast_results")
    if forecast:
        print(f"Forecast Agent Output:")
        print(f"- Forecast Predictions count: {len(forecast.get('predictions', []))}")
        print(f"- Forecast Chart saved to: {forecast.get('chart_path')}")
        
    explanation = result_state.get("explanation_results")
    if explanation:
        print(f"Explanation Agent Summary (length): {len(explanation)} chars")
        
    recommendations = result_state.get("recommendation_results")
    if recommendations:
        print(f"Recommendations Agent Summary (length): {len(recommendations)} chars")
        
    final_report = result_state.get("final_report")
    if final_report:
        print(f"Final Compiled Report generated. Size: {len(final_report)} chars")
        
    return True

if __name__ == "__main__":
    rag_ok = test_rag_retrieval()
    if rag_ok:
        orchestrator_ok = test_orchestrator_pipeline()
        if orchestrator_ok:
            print("\nAll automated verification checks passed!")
        else:
            print("\nOrchestrator pipeline verification check failed.")
    else:
        print("\nRAG retrieval verification check failed.")
