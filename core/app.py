# FastAPI core wrapper
import os, time, threading
from fastapi import FastAPI
from pydantic import BaseModel
from DeterministicQueryEngine import DeterministicQueryEngine, ENGINE_VERSION
from pathlib import Path

DATA_PATH = os.getenv("DATA_PATH", "data/standardized_dqs_full.json")
if not Path(DATA_PATH).exists():
    DATA_PATH = "standardized_dqs_full.json"

app = FastAPI()

# Global state
engine = None
loading = True

class Q(BaseModel):
    query: str

def load_engine():
    """Load engine in background thread"""
    global engine, loading
    try:
        engine = DeterministicQueryEngine(DATA_PATH)
        loading = False
        print("✅ Engine loaded successfully", flush=True)
    except Exception as e:
        print(f"❌ Engine loading failed: {e}", flush=True)
        loading = False

# Start loading in background immediately
threading.Thread(target=load_engine, daemon=True).start()

@app.get("/")
def root():
    if loading:
        return {"status": "initializing", "service": "DQS Core API", "version": ENGINE_VERSION}
    return {"status": "ok", "service": "DQS Core API", "version": ENGINE_VERSION, "ready": engine is not None}

@app.get("/health")
def health():
    if loading or engine is None:
        return {
            "status": "initializing" if loading else "error",
            "engine_version": ENGINE_VERSION,
            "master_data_loaded": False,
            "rows": 0
        }
    
    df_rows = int(engine.master_df.shape[0]) if engine.master_df is not None else 0
    return {
        "status": "ok",
        "engine_version": ENGINE_VERSION,
        "master_data_loaded": engine.master_df is not None,
        "rows": df_rows,
        "data_path": DATA_PATH,
        "strict_topic_only": engine.strict_topic_only,
        "strict_no_implicit_group": engine.strict_no_implicit_group,
        "config_files_loaded": {
            "keyword_mappings": bool(engine.keyword_mappings.get("topic_mappings")),
            "topic_links": bool(engine.topic_links),
            "group_label_map": bool(engine.group_label_map),
            "dataset_overrides": bool(engine.dataset_overrides),
        }
    }

@app.post("/v1/widget_text")
def widget_text(payload: Q):
    if loading or engine is None:
        return {"error": "Service initializing, please try again in a moment"}
    
    t0 = time.time()
    res = engine.query(payload.query)
    ms = int((time.time() - t0) * 1000)
    return {
        "engine_version": ENGINE_VERSION,
        "ms_elapsed": ms,
        "answer_text": res.summary,
        "topic": res.topic,
        "dataset_id": res.dataset_id,
        "group": res.group,
        "years": res.years,
        "warnings": res.warnings
    }
