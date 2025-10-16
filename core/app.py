# FastAPI core wrapper
import os, time
from fastapi import FastAPI
from pydantic import BaseModel
from DeterministicQueryEngine import DeterministicQueryEngine, ENGINE_VERSION
from pathlib import Path

DATA_PATH = os.getenv("DATA_PATH", "data/standardized_dqs_full.json")
if not Path(DATA_PATH).exists():
    # fallback to repo root
    DATA_PATH = "standardized_dqs_full.json"

engine = DeterministicQueryEngine(DATA_PATH)
app = FastAPI()

class Q(BaseModel):
    query: str

@app.get("/")
def root():
    return {"status": "ok", "service": "DQS Core API", "version": ENGINE_VERSION}

@app.get("/health")
def health():
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
