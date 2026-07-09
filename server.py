import os
import shutil
import json
from datetime import datetime
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Import orchestrator and rag_builder
from orchestrator import bi_workflow, call_llm
from rag_builder import FAISSVectorStore, build_vector_store_from_csv

app = FastAPI(title="AI Business Intelligence Multi-Agent System")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Add custom cache control middleware to prevent browser caching of frontend files during development
@app.middleware("http")
async def add_cache_control_header(request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.startswith("/static"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

DATASET_DIR = "datasets"
FALLBACK_FILENAME = "Sample - Superstore1.csv"
FAISS_INDEX_DIR = "faiss_index"
CONFIG_PATH = os.path.join(DATASET_DIR, "active_config.json")

os.makedirs(DATASET_DIR, exist_ok=True)

def set_active_dataset(filename):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({"active_filename": filename}, f)
    except Exception as e:
        print(f"Error saving active dataset configuration: {e}")

def get_dataset_path():
    # Read configuration
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                config = json.load(f)
                filename = config.get("active_filename")
                if filename:
                    filepath = os.path.join(DATASET_DIR, filename)
                    if os.path.exists(filepath):
                        return filepath
        except Exception:
            pass
            
    # Fallback: scan datasets/ folder
    csv_files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.csv')]
    if csv_files:
        filepath = os.path.join(DATASET_DIR, csv_files[0])
        set_active_dataset(csv_files[0])
        return filepath
        
    # Fallback to root Sample - Superstore1.csv
    if os.path.exists(FALLBACK_FILENAME):
        return FALLBACK_FILENAME
        
    return os.path.join(DATASET_DIR, "dataset.csv")

# Create static folder for frontend and static assets
os.makedirs("static", exist_ok=True)

# Pydantic schemas
class AnalyzeRequest(BaseModel):
    question: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    question: str
    history: List[ChatMessage]

class FeedbackRequest(BaseModel):
    rating: int
    category: str
    comment: str

@app.get("/api/status")
def get_status():
    """Returns the ingestion status of the dataset and FAISS database."""
    dataset_path = get_dataset_path()
    dataset_exists = os.path.exists(dataset_path)
    faiss_exists = os.path.exists(os.path.join(FAISS_INDEX_DIR, "index.faiss"))
    
    dataset_name = None
    if dataset_exists:
        dataset_name = os.path.basename(dataset_path)
        
    return {
        "dataset_loaded": dataset_exists,
        "dataset_name": dataset_name,
        "faiss_indexed": faiss_exists,
        "api_key_configured": os.getenv("GROQ_API_KEY") is not None
    }

@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Uploads a new Superstore CSV dataset and builds the FAISS vector index."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        
    try:
        os.makedirs(DATASET_DIR, exist_ok=True)
        # Prevent path injection
        filename = os.path.basename(file.filename)
        temp_path = os.path.join(DATASET_DIR, filename)
        
        # Save file to datasets folder
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        print(f"Dataset saved to {temp_path}. Rebuilding FAISS vector database...")
        
        # Set active config
        set_active_dataset(filename)
        
        # Build index
        success = build_vector_store_from_csv(temp_path, FAISS_INDEX_DIR)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to build FAISS index from dataset.")
            
        return {
            "status": "success",
            "message": "Dataset uploaded and FAISS vector index built successfully."
        }
    except Exception as e:
        print(f"Error in dataset upload: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/api/datasets")
def list_datasets():
    """Returns list of uploaded datasets and the active file."""
    os.makedirs(DATASET_DIR, exist_ok=True)
    files = []
    active_path = get_dataset_path()
    active_filename = os.path.basename(active_path) if active_path else None
    
    # Check if fallback exists
    if os.path.exists(FALLBACK_FILENAME):
        stat = os.stat(FALLBACK_FILENAME)
        files.append({
            "filename": FALLBACK_FILENAME,
            "size": stat.st_size,
            "uploaded_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "active": (active_path == FALLBACK_FILENAME)
        })
        
    for f in os.listdir(DATASET_DIR):
        if f.endswith('.csv') and f != "active_config.json":
            filepath = os.path.join(DATASET_DIR, f)
            stat = os.stat(filepath)
            files.append({
                "filename": f,
                "size": stat.st_size,
                "uploaded_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "active": (f == active_filename and active_path != FALLBACK_FILENAME)
            })
            
    # Deduplicate
    seen = set()
    dedup_files = []
    for file_info in files:
        if file_info["filename"] not in seen:
            seen.add(file_info["filename"])
            dedup_files.append(file_info)
            
    return dedup_files

class SelectDatasetRequest(BaseModel):
    filename: str

@app.post("/api/datasets/select")
def select_dataset(req: SelectDatasetRequest):
    """Sets active dataset and rebuilds the FAISS index."""
    filename = req.filename
    if filename == FALLBACK_FILENAME and os.path.exists(FALLBACK_FILENAME):
        filepath = FALLBACK_FILENAME
        # Remove custom active config to fallback
        if os.path.exists(CONFIG_PATH):
            try:
                os.remove(CONFIG_PATH)
            except Exception:
                pass
    else:
        filepath = os.path.join(DATASET_DIR, filename)
        if not os.path.exists(filepath):
            raise HTTPException(status_code=404, detail=f"File {filename} not found.")
        set_active_dataset(filename)
        
    # Rebuild index
    print(f"Dataset selected: {filepath}. Rebuilding FAISS vector database...")
    success = build_vector_store_from_csv(filepath, FAISS_INDEX_DIR)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to build FAISS index from selected dataset.")
        
    return {"status": "success", "message": f"Activated {filename} and rebuilt index successfully."}

@app.delete("/api/datasets/{filename}")
def delete_dataset(filename: str):
    """Deletes a dataset from history."""
    if filename == FALLBACK_FILENAME:
        raise HTTPException(status_code=400, detail="Cannot delete system fallback dataset.")
        
    filepath = os.path.join(DATASET_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found.")
        
    active_path = get_dataset_path()
    if os.path.basename(active_path) == filename and active_path != FALLBACK_FILENAME:
        if os.path.exists(CONFIG_PATH):
            try:
                os.remove(CONFIG_PATH)
            except Exception:
                pass
                
    try:
        os.remove(filepath)
        return {"status": "success", "message": f"Deleted {filename} successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")

@app.get("/api/dataset/preview")
def preview_dataset():
    """Returns first 100 rows of the active dataset for frontend display."""
    filepath = get_dataset_path()
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="No active dataset loaded.")
        
    try:
        try:
            df = pd.read_csv(filepath, encoding="latin1")
        except Exception:
            df = pd.read_csv(filepath, encoding="utf-8")
            
        df_preview = df.fillna("")
        # Only return first 100 rows
        return df_preview.head(100).to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read dataset: {str(e)}")

@app.post("/api/analyze")
def run_analysis_pipeline(req: AnalyzeRequest):
    """Triggers the LangGraph multi-agent orchestrator for BI reporting."""
    dataset_path = get_dataset_path()
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=400, detail="Dataset not loaded. Please upload a dataset first.")
        
    try:
        initial_state = {
            "question": req.question,
            "execution_plan": [],
            "history": [],
            "dataset_path": dataset_path,
            "analysis_results": None,
            "root_cause_results": None,
            "forecast_results": None,
            "explanation_results": None,
            "recommendation_results": None,
            "final_report": None,
            "error": None
        }
        
        print(f"Invoking BI workflow graph for question: '{req.question}'")
        output_state = bi_workflow.invoke(initial_state)
        
        if output_state.get("error"):
            raise HTTPException(status_code=500, detail=output_state["error"])
            
        return output_state
    except Exception as e:
        print(f"Error executing analysis graph: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")

@app.post("/api/chat")
def chat_rag(req: ChatRequest):
    """Chat Agent that uses Retrieval-Augmented Generation (RAG) and session memory."""
    dataset_path = get_dataset_path()
    if not os.path.exists(dataset_path):
        raise HTTPException(status_code=400, detail="Dataset not loaded. Please upload a dataset first.")
        
    store = FAISSVectorStore(FAISS_INDEX_DIR)
    if not store.load_index():
        raise HTTPException(status_code=400, detail="FAISS vector index not initialized. Please build/upload first.")
        
    try:
        # 1. Retrieve context
        print(f"RAG Chat: Retrieving facts for query '{req.question}'")
        retrieved_facts = store.search(req.question, k=5)
        
        # Format context and history
        context_str = "\n".join([f"- {fact}" for fact in retrieved_facts])
        
        history_str = ""
        for msg in req.history:
            history_str += f"{msg.role.capitalize()}: {msg.content}\n"
            
        prompt = f"""
        You are a Senior Business Intelligence chatbot.
        Answer the user's business question using ONLY the provided business facts from our data catalog.
        
        Rules:
        1. Base your answers strictly on the facts listed in the context.
        2. If the context does not contain the answer, say "I don't have that specific data in my current analysis, but I can help you analyze the dataset if you tell me what you're looking for."
        3. Do not make up any numbers or insights.
        4. Refer to the specific numbers (e.g. Sales, Profits, Discounts) in your response.
        
        Retrieved Business Facts (Context):
        {context_str if context_str else "No relevant facts found."}
        
        Conversation History:
        {history_str}
        
        User Question: {req.question}
        
        Business Answer:
        """
        
        # 2. Call LLM to generate answer
        response_content = call_llm(prompt)
        
        # 3. Format response
        updated_history = [m.dict() for m in req.history]
        updated_history.append({"role": "user", "content": req.question})
        updated_history.append({"role": "assistant", "content": response_content})
        
        return {
            "answer": response_content,
            "history": updated_history,
            "references": retrieved_facts
        }
    except Exception as e:
        print(f"Error in RAG chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Chat execution failed: {str(e)}")

FEEDBACK_FILE = "feedback.json"

@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest):
    """Saves user feedback to a local JSON file."""
    feedback_entry = {
        "rating": req.rating,
        "category": req.category,
        "comment": req.comment,
        "timestamp": datetime.now().isoformat()
    }
    
    feedback_list = []
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                feedback_list = json.load(f)
                if not isinstance(feedback_list, list):
                    feedback_list = []
        except Exception:
            feedback_list = []
            
    feedback_list.append(feedback_entry)
    
    try:
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(feedback_list, f, indent=4, ensure_ascii=False)
        return {"status": "success", "message": "Feedback submitted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")

@app.get("/api/feedback")
def get_feedback():
    """Retrieves all feedback entries."""
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

# Mount static files folder to serve the frontend React SPA
app.mount("/static", StaticFiles(directory="static"), name="static")

# Redirect root path to index.html
@app.get("/")
def read_root():
    from fastapi.responses import FileResponse
    index_file = "static/index.html"
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "API is online. Please create static/index.html to view the dashboard."}
