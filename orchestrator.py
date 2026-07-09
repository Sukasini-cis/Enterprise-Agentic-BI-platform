import os
import json
import pandas as pd
import numpy as np
import matplotlib
# Use non-interactive backend for Matplotlib to avoid GUI errors in background tasks
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prophet import Prophet
from groq import Groq
from typing import Dict, List, Any, Optional, TypedDict
from langgraph.graph import StateGraph, END

# Define LangGraph State Schema
class AgentState(TypedDict):
    question: str
    execution_plan: List[str]
    history: List[Dict[str, str]]
    dataset_path: str
    analysis_results: Optional[Dict[str, Any]]
    root_cause_results: Optional[Dict[str, Any]]
    forecast_results: Optional[Dict[str, Any]]
    explanation_results: Optional[str]
    recommendation_results: Optional[str]
    final_report: Optional[str]
    error: Optional[str]

# Initialize Groq client
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Warning: GROQ_API_KEY not found in environment. Using simulated AI responses.")
        return None
    try:
        return Groq(api_key=api_key)
    except Exception as e:
        print(f"Error initializing Groq client: {e}")
        return None

# LLM Helper function with mock fallback
def call_llm(prompt: str, model_name: str = "llama-3.1-8b-instant") -> str:
    client = get_groq_client()
    if not client:
        return get_simulated_response(prompt)
        
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Groq API call failed: {e}. Falling back to simulation.")
        return get_simulated_response(prompt)

def get_simulated_response(prompt: str) -> str:
    """Simulates realistic BI insights if Groq is unavailable."""
    prompt_lower = prompt.lower()
    if "planner" in prompt_lower:
        return '["analysis", "root_cause", "forecast", "explanation", "recommendation"]'
    elif "forecast" in prompt_lower or "explanation" in prompt_lower:
        return """### Executive Sales Forecast Explanation
- **Sales Trajectory**: The forecast indicates a steady growth rate of approximately 4.5% month-over-month.
- **Seasonality & Peaks**: Peak revenue is anticipated in November and December, aligned with historical Q4 retail surges.
- **Critical Risks**: A widening confidence interval in the final months suggests higher uncertainty, likely due to fluctuating regional discount trends.
- **Action Items**: Secure inventory ahead of the September-October ramp-up and monitor discount margins."""
    elif "recommend" in prompt_lower:
        return """### Executive Business Recommendations

#### 1. Strategic Pricing and Discount Policy
- **Priority**: High
- **Action**: Implement a discount cap of 15% on Tables and Bookcases.
- **Expected Impact**: Standardizing the margin profile will reclaim approximately $21,000 in lost profitability.

#### 2. Regional Re-allocation
- **Priority**: Medium
- **Action**: Direct marketing spend away from low-margin Central segments into high-margin West region tech categories.
- **Expected Impact**: Elevates overall margin from 12.4% to 14.5% within 2 quarters.

#### 3. Risk Mitigation
- **Priority**: Medium
- **Action**: Renegotiate supply-chain freight costs for heavy furniture lines (Tables) to offset shipping costs.
- **Expected Impact**: Reduces the cost of goods sold (COGS) for unprofitable products."""
    else:
        return "Insight: Performance shows typical retail seasonality. Attention is required on high discount margins in furniture products."

# --- NODE 1: Planner Agent ---
def planner_node(state: AgentState) -> Dict[str, Any]:
    print("\n--- RUNNING PLANNER AGENT ---")
    question = state.get("question", "")
    
    prompt = f"""
    You are an Agentic Business Intelligence system planner.
    Analyze the user's question and decide which agents need to execute and in what order.
    
    Available agents:
    - analysis: Computes KPIs (Sales, Profit, margins) and descriptive category/regional breakdowns.
    - root_cause: Identifies loss-making products, categories, high discounts, and explains root causes of poor performance.
    - forecast: Predicts future sales and generates forecast trends and graphs.
    - explanation: Translates forecast predictions and trends into business insights.
    - recommendation: Synthesizes analysis, root causes, and forecasting to generate business recommendations, growth opportunities, risks, and priority actions.
    
    If the question is general or requires a comprehensive business review, or asks about declining profit/sales performance, schedule all agents in sequence:
    ["analysis", "root_cause", "forecast", "explanation", "recommendation"]
    
    If the question is specifically about predicting, forecasting, or future sales, schedule:
    ["forecast", "explanation"]
    
    If the question is about descriptive performance (sales/profits today), schedule:
    ["analysis"]
    
    If the question is about why we are losing money or where performance is poor, schedule:
    ["analysis", "root_cause"]
    
    Return ONLY a JSON array of the agent names in the order they should run. Do not include any markdown styling, explanation, or extra text.
    Example Output: ["analysis", "root_cause", "forecast", "explanation", "recommendation"]
    
    User Question: {question}
    """
    
    output_text = call_llm(prompt).strip()
    # Strip markdown if present
    if output_text.startswith("```json"):
        output_text = output_text[7:]
    if output_text.endswith("```"):
        output_text = output_text[:-3]
    output_text = output_text.strip()
    
    try:
        execution_plan = json.loads(output_text)
        if not isinstance(execution_plan, list):
            execution_plan = ["analysis", "root_cause", "forecast", "explanation", "recommendation"]
    except Exception as e:
        print(f"Failed to parse planner plan: {e}. Defaulting to full sequence.")
        execution_plan = ["analysis", "root_cause", "forecast", "explanation", "recommendation"]
        
    print(f"Execution Plan: {execution_plan}")
    return {"execution_plan": execution_plan}

# --- NODE 2: Analysis Agent ---
def analysis_node(state: AgentState) -> Dict[str, Any]:
    print("\n--- RUNNING ANALYSIS AGENT ---")
    csv_path = state.get("dataset_path")
    if not csv_path or not os.path.exists(csv_path):
        return {"error": f"Dataset path not found: {csv_path}"}
        
    try:
        try:
            df = pd.read_csv(csv_path, encoding="latin1")
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding="utf-8")
            
        df.columns = [c.strip() for c in df.columns]
        df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0)
        df["Profit"] = pd.to_numeric(df["Profit"], errors="coerce").fillna(0)
        df["Discount"] = pd.to_numeric(df["Discount"], errors="coerce").fillna(0)
        
        # Calculate main KPIs
        total_sales = float(df["Sales"].sum())
        total_profit = float(df["Profit"].sum())
        overall_margin = float((total_profit / total_sales) * 100) if total_sales > 0 else 0.0
        avg_discount = float(df["Discount"].mean() * 100)
        
        # Regional Breakdown
        reg_df = df.groupby("Region").agg({"Sales": "sum", "Profit": "sum"}).reset_index()
        regional_analysis = []
        for _, r in reg_df.iterrows():
            regional_analysis.append({
                "region": r["Region"],
                "sales": float(r["Sales"]),
                "profit": float(r["Profit"]),
                "margin": float((r["Profit"] / r["Sales"]) * 100) if r["Sales"] > 0 else 0
            })
            
        # Category Breakdown
        cat_df = df.groupby("Category").agg({"Sales": "sum", "Profit": "sum", "Discount": "mean"}).reset_index()
        category_analysis = []
        for _, r in cat_df.iterrows():
            category_analysis.append({
                "category": r["Category"],
                "sales": float(r["Sales"]),
                "profit": float(r["Profit"]),
                "margin": float((r["Profit"] / r["Sales"]) * 100) if r["Sales"] > 0 else 0,
                "discount": float(r["Discount"] * 100)
            })
            
        # Sub-category Breakdown
        sub_df = df.groupby(["Category", "Sub-Category"]).agg({"Sales": "sum", "Profit": "sum", "Discount": "mean"}).reset_index()
        subcategory_analysis = []
        for _, r in sub_df.iterrows():
            subcategory_analysis.append({
                "category": r["Category"],
                "subcategory": r["Sub-Category"],
                "sales": float(r["Sales"]),
                "profit": float(r["Profit"]),
                "margin": float((r["Profit"] / r["Sales"]) * 100) if r["Sales"] > 0 else 0,
                "discount": float(r["Discount"] * 100)
            })
            
        analysis_results = {
            "total_sales": total_sales,
            "total_profit": total_profit,
            "overall_margin": overall_margin,
            "avg_discount": avg_discount,
            "regional_analysis": regional_analysis,
            "category_analysis": category_analysis,
            "subcategory_analysis": subcategory_analysis
        }
        
        print("Analysis completed successfully.")
        return {"analysis_results": analysis_results}
    except Exception as e:
        print(f"Error in analysis_node: {e}")
        return {"error": f"Analysis failed: {str(e)}"}

# --- NODE 3: Root Cause Agent ---
def root_cause_node(state: AgentState) -> Dict[str, Any]:
    print("\n--- RUNNING ROOT CAUSE AGENT ---")
    csv_path = state.get("dataset_path")
    if not csv_path or not os.path.exists(csv_path):
        return {"error": "Dataset not found."}
        
    try:
        try:
            df = pd.read_csv(csv_path, encoding="latin1")
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding="utf-8")
            
        df.columns = [c.strip() for c in df.columns]
        df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0)
        df["Profit"] = pd.to_numeric(df["Profit"], errors="coerce").fillna(0)
        df["Discount"] = pd.to_numeric(df["Discount"], errors="coerce").fillna(0)
        
        # 1. Worst Performing Sub-Categories
        sub_perf = df.groupby("Sub-Category").agg({"Profit": "sum", "Sales": "sum", "Discount": "mean"}).reset_index()
        loss_subcats = sub_perf[sub_perf["Profit"] < 0].sort_values(by="Profit").to_dict(orient="records")
        for x in loss_subcats:
            x["Profit"] = float(x["Profit"])
            x["Sales"] = float(x["Sales"])
            x["Discount"] = float(x["Discount"] * 100)
            
        # 2. Top Loss-Making Products
        prod_perf = df.groupby(["Sub-Category", "Product Name"]).agg({"Profit": "sum", "Sales": "sum", "Discount": "mean"}).reset_index()
        loss_products = prod_perf[prod_perf["Profit"] < 0].sort_values(by="Profit").head(10).to_dict(orient="records")
        for x in loss_products:
            x["Profit"] = float(x["Profit"])
            x["Sales"] = float(x["Sales"])
            x["Discount"] = float(x["Discount"] * 100)
            
        # 3. Correlation calculation
        corr_discount_profit = float(df["Discount"].corr(df["Profit"]))
        
        # Format Root Cause Analysis Report using Groq
        loss_summary = ""
        for i, item in enumerate(loss_subcats):
            loss_summary += f"- {item['Sub-Category']}: Profit: ${item['Profit']:,.2f}, Sales: ${item['Sales']:,.2f}, Avg Discount: {item['Discount']:.1f}%\n"
            
        prod_summary = ""
        for i, item in enumerate(loss_products[:5]):
            prod_summary += f"- {item['Product Name']} ({item['Sub-Category']}): Profit: ${item['Profit']:,.2f}, Sales: ${item['Sales']:,.2f}, Avg Discount: {item['Discount']:.1f}%\n"
            
        prompt = f"""
        You are a Root Cause Analysis Agent.
        Analyze these underperforming areas in our sales dataset:
        
        Loss-making Sub-categories:
        {loss_summary if loss_summary else "None"}
        
        Top Loss-making Products:
        {prod_summary if prod_summary else "None"}
        
        Correlation between Discount and Profit: {corr_discount_profit:.4f}
        
        Explain:
        1. Why these specific categories/products are losing money.
        2. The role discount is playing in these losses (e.g., discount erosion).
        3. Make it clear, concise, and formatted in markdown.
        """
        
        explanation = call_llm(prompt)
        
        root_cause_results = {
            "loss_subcategories": loss_subcats,
            "loss_products": loss_products,
            "correlation_discount_profit": corr_discount_profit,
            "explanation": explanation
        }
        
        print("Root cause analysis completed.")
        return {"root_cause_results": root_cause_results}
    except Exception as e:
        print(f"Error in root_cause_node: {e}")
        return {"error": f"Root Cause Agent failed: {str(e)}"}

import threading
matplotlib_lock = threading.Lock()

# --- NODE 4: Forecast Agent ---
def forecast_node(state: AgentState) -> Dict[str, Any]:
    print("\n--- RUNNING FORECAST AGENT ---")
    csv_path = state.get("dataset_path")
    if not csv_path or not os.path.exists(csv_path):
        return {"error": "Dataset not found."}
        
    try:
        try:
            df = pd.read_csv(csv_path, encoding="latin1")
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding="utf-8")
            
        df.columns = [c.strip() for c in df.columns]
        df["Order Date"] = pd.to_datetime(df["Order Date"], format="mixed")
        df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce").fillna(0)
        
        # Aggregate sales by month
        monthly_sales = df.groupby(pd.Grouper(key="Order Date", freq="ME"))["Sales"].sum().reset_index()
        monthly_sales.columns = ["ds", "y"]
        
        # Fit Prophet Model
        # Quiet the logs of Prophet
        import logging
        logging.getLogger('prophet').setLevel(logging.WARNING)
        
        model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        model.fit(monthly_sales)
        
        # Forecast 6 periods (months)
        future = model.make_future_dataframe(periods=6, freq="ME")
        forecast = model.predict(future)
        
        # Create static folder if it doesn't exist
        os.makedirs("static", exist_ok=True)
        
        # Generate plot safely using thread lock to prevent matplotlib deadlocks in web servers
        with matplotlib_lock:
            fig = model.plot(forecast)
            plt.title("Sales Forecast & Trend (FB Prophet)")
            plt.xlabel("Timeline")
            plt.ylabel("Monthly Revenue ($)")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.savefig("static/forecast.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        # Format the forecast output
        forecast_tail = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(6)
        predictions = []
        for _, row in forecast_tail.iterrows():
            predictions.append({
                "date": row["ds"].strftime("%Y-%m-%d"),
                "forecast": float(row["yhat"]),
                "lower": float(row["yhat_lower"]),
                "upper": float(row["yhat_upper"])
            })
            
        # Get historical monthly values for UI plotting
        history_df = monthly_sales.tail(12)
        history = []
        for _, row in history_df.iterrows():
            history.append({
                "date": row["ds"].strftime("%Y-%m-%d"),
                "sales": float(row["y"])
            })
            
        print("Prophet sales forecast completed successfully.")
        return {
            "forecast_results": {
                "predictions": predictions,
                "history": history,
                "chart_path": "/static/forecast.png"
            }
        }
    except Exception as e:
        print(f"Error in forecast_node: {e}")
        return {"error": f"Forecast Agent failed: {str(e)}"}

# --- NODE 5: Explanation Agent ---
def explanation_node(state: AgentState) -> Dict[str, Any]:
    print("\n--- RUNNING EXPLANATION AGENT ---")
    forecast_results = state.get("forecast_results")
    if not forecast_results:
        return {"error": "Forecast data not available for explanation."}
        
    predictions = forecast_results.get("predictions", [])
    forecast_text = ""
    for p in predictions:
        forecast_text += f"- Date: {p['date']}, Forecasted Sales: ${p['forecast']:,.2f} (Confidence: ${p['lower']:,.2f} to ${p['upper']:,.2f})\n"
        
    prompt = f"""
    You are a Senior Business Analyst.
    Analyze the following sales forecast results:
    
    Forecasted Sales for the Next 6 Months:
    {forecast_text}
    
    Provide:
    1. Overall Revenue Trend (increasing, decreasing, stable).
    2. Best anticipated month and lowest anticipated month.
    3. Potential business risks indicated by the confidence intervals or historical data.
    4. Practical, high-level business insights.
    
    Format in clean markdown for an executive audience.
    """
    
    explanation = call_llm(prompt)
    print("Forecast explanation completed.")
    return {"explanation_results": explanation}

# --- NODE 6: Recommendation Agent ---
def recommendation_node(state: AgentState) -> Dict[str, Any]:
    print("\n--- RUNNING RECOMMENDATION AGENT ---")
    analysis = state.get("analysis_results", {})
    root_cause = state.get("root_cause_results", {})
    explanation = state.get("explanation_results", "")
    
    # Format business metrics summary
    business_summary = f"""
    Business KPIs:
    - Total Sales: ${analysis.get('total_sales', 0):,.2f}
    - Total Profit: ${analysis.get('total_profit', 0):,.2f}
    - Profit Margin: {analysis.get('overall_margin', 0):.2f}%
    - Average Discount: {analysis.get('avg_discount', 0):.2f}%
    """
    if root_cause:
        business_summary += f"\nRoot Cause: Underperforming sub-categories show average discounts like:\n"
        for item in root_cause.get("loss_subcategories", []):
            business_summary += f"- {item['Sub-Category']}: Loss: ${item['Profit']:,.2f}, Discount: {item['Discount']:.1f}%\n"
            
    prompt = f"""
    You are a Lead Business Consultant.
    Review the following business data, root cause explanation, and sales forecast insights:
    
    {business_summary}
    
    Forecast Insights:
    {explanation}
    
    Provide:
    1. Executive Summary: Quick overview of state of the business.
    2. Priority Action Items: Clear, bulleted action items with assigned priorities (High, Medium, Low).
    3. Growth Opportunities: Where to invest or allocate capital based on profitable segments/regions.
    4. Risk Mitigation Strategies: How to fix the loss-making segments or pricing structures.
    5. Final Business Decision/Verdict.
    
    Format the response in clean, professional markdown.
    """
    
    recommendation = call_llm(prompt)
    
    # Save the final report text
    final_report = f"""# Executive Business Report

## 1. System Performance Overview
{business_summary}

## 2. Root Cause Analysis
{root_cause.get('explanation', 'No details available.') if root_cause else 'No details available.'}

## 3. Sales Forecasting & Analysis
{explanation}

## 4. Strategic Recommendations & Action Plan
{recommendation}
"""
    # Also write to recommendations.txt for persistence
    try:
        with open("recommendations.txt", "w", encoding="utf-8") as f:
            f.write(final_report)
    except Exception as e:
        print(f"Error writing recommendations.txt: {e}")
        
    print("Recommendations agent completed.")
    return {
        "recommendation_results": recommendation,
        "final_report": final_report
    }

# --- LANGGRAPH ROUTER FUNCTION ---
def route_agents(state: AgentState) -> str:
    if state.get("error"):
        print(f"Workflow encountered error: {state.get('error')}. Ending.")
        return END
        
    plan = state.get("execution_plan", [])
    if not plan:
        print("No execution plan found. Ending.")
        return END
        
    # Check for the next uncompleted agent
    if "analysis" in plan and not state.get("analysis_results"):
        return "analysis"
    elif "root_cause" in plan and not state.get("root_cause_results"):
        return "root_cause"
    elif "forecast" in plan and not state.get("forecast_results"):
        return "forecast"
    elif "explanation" in plan and not state.get("explanation_results"):
        return "explanation"
    elif "recommendation" in plan and not state.get("recommendation_results"):
        return "recommendation"
        
    print("All scheduled agents executed successfully. Ending.")
    return END

# --- COMPILE GRAPH ---
def build_bi_workflow() -> StateGraph:
    builder = StateGraph(AgentState)
    
    # Add Nodes
    builder.add_node("planner", planner_node)
    builder.add_node("analysis", analysis_node)
    builder.add_node("root_cause", root_cause_node)
    builder.add_node("forecast", forecast_node)
    builder.add_node("explanation", explanation_node)
    builder.add_node("recommendation", recommendation_node)
    
    # Set Entry Point
    builder.set_entry_point("planner")
    
    # Setup Conditional Routing Mapping
    mapping = {
        "analysis": "analysis",
        "root_cause": "root_cause",
        "forecast": "forecast",
        "explanation": "explanation",
        "recommendation": "recommendation",
        END: END
    }
    
    # Add Edges
    builder.add_conditional_edges("planner", route_agents, mapping)
    builder.add_conditional_edges("analysis", route_agents, mapping)
    builder.add_conditional_edges("root_cause", route_agents, mapping)
    builder.add_conditional_edges("forecast", route_agents, mapping)
    builder.add_conditional_edges("explanation", route_agents, mapping)
    builder.add_conditional_edges("recommendation", route_agents, mapping)
    
    return builder.compile()

# Instantiate compiled workflow
bi_workflow = build_bi_workflow()

if __name__ == "__main__":
    # Test workflow locally if dataset exists
    csv_file = "Sample - Superstore1.csv"
    if os.path.exists(csv_file):
        test_state = {
            "question": "Generate a full business analysis and forecast sales for next six months.",
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
        print("Starting test execution of BI Workflow...")
        result = bi_workflow.invoke(test_state)
        if result.get("error"):
            print(f"Test failed with error: {result['error']}")
        else:
            print("\nWorkflow completed successfully!")
            print(f"Planner output: {result.get('execution_plan')}")
            print(f"Overall sales computed: ${result.get('analysis_results', {}).get('total_sales', 0):,.2f}")
            print(f"Forecasted predictions: {len(result.get('forecast_results', {}).get('predictions', []))} records generated.")
            print(f"Final Report size: {len(result.get('final_report', ''))} characters.")
    else:
        print(f"Test CSV '{csv_file}' not found.")
