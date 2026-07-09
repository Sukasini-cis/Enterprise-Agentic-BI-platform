
# Enterprise Autonomous Multi-Agent BI Platform

A state-of-the-art Business Intelligence command center that orchestrates automated planning, time-series forecasting, root-cause diagnostics, and context-aware semantic knowledge base searches.

Built using **FastAPI** for the backend, **React 18** for the frontend, **LangGraph** for multi-agent workflows, **FB Prophet** for machine learning forecasts, and **FAISS** for vector embedding search.

---

## 🚀 Key Features

1. **Autonomous Multi-Agent Intelligence**: Coordinates a 6-agent pipeline (Planner → Descriptive Analytics → Diagnostic Engine → Trend Forecasting → Explanation Agent → Strategic Insights) to generate executive-level business analyses.
2. **Interactive Dataset Grid & CSV Explorer**: Inspect uploaded CSVs directly in the interface. Sort columns dynamically, run keyword searches across cells, and filter rows by Category or Region.
3. **Executive PDF Exporter**: Export comprehensive reports to print or PDF with optimized page-break rules and high-contrast styling.
4. **AI Knowledge Chat (RAG)**: Chat with your dataset to extract instant context-aware business answers retrieved from the local FAISS semantic index.
5. **Theme Personalization**: Toggle between a dark aesthetic or a high-contrast light workspace theme.

---

## 🛠️ File Structure

* `server.py`: FastAPI server containing CSV preview, feedback, multi-agent analysis, and RAG chat API routes.
* `orchestrator.py`: Multi-agent pipeline defining graph execution paths and calling LLM APIs.
* `rag_builder.py`: Embedded fact builder that generates local FAISS vector store indexes.
* `static/index.html`: Modern, single-page React client application.
* `requirements.txt`: Python library dependencies.
* `.gitignore`: Excludes local data caches, model files, and system files.

---

## ⚙️ Setup & Installation

### 1. Clone the Project
```bash
git clone <your-repository-url>
cd <repository-folder>
```

### 2. Install Dependencies
Ensure you have Python 3.10+ installed:
```bash
pip install -r requirements.txt
```

### 3. Set Up API Key
Configure your Groq API Key environment variable:
```bash
# Windows (PowerShell)
$env:GROQ_API_KEY="your_groq_api_key_here"

# Windows (Command Prompt)
set GROQ_API_KEY="your_groq_api_key_here"
```

### 4. Run the Platform
Start the FastAPI server:
```bash
python -m uvicorn server:app --reload --port 8000
```
Open your browser and navigate to **[http://localhost:8000](http://localhost:8000)**. Use the credentials `admin@enterprise.bi` / `password123` to access the workspace.
