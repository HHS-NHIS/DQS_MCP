# FastAPI core wrapper
import os, time
from fastapi import FastAPI, Response, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from DeterministicQueryEngine import DeterministicQueryEngine, ENGINE_VERSION
from pathlib import Path
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

DATA_PATH = os.getenv("DATA_PATH", "data/standardized_dqs_full.json")
if not Path(DATA_PATH).exists():
    DATA_PATH = "standardized_dqs_full.json"

print(f"🔄 Loading engine from {DATA_PATH}...", flush=True)
engine = DeterministicQueryEngine(DATA_PATH)
print(f"✅ Engine loaded with {engine.master_df.shape[0] if engine.master_df is not None else 0} rows", flush=True)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address, default_limits=["200/hour"])

app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

class Q(BaseModel):
    query: str

@app.get("/")
@limiter.limit("60/minute")
def root(request: Request):
    return {
        "status": "ok",
        "service": "DQS Core API",
        "version": ENGINE_VERSION,
        "ready": True
    }

@app.get("/health")
@limiter.limit("60/minute")
def health(request: Request):
    if engine is None or engine.master_df is None:
        return Response(
            content='{"status":"error","message":"Engine not loaded"}',
            status_code=503,
            media_type="application/json"
        )
    
    df_rows = int(engine.master_df.shape[0])
    return {
        "status": "ok",
        "engine_version": ENGINE_VERSION,
        "master_data_loaded": True,
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
@limiter.limit("30/minute")  # 30 queries per minute per IP
def widget_text(request: Request, payload: Q):
    if engine is None or engine.master_df is None:
        return Response(
            content='{"error":"Service not ready"}',
            status_code=503,
            media_type="application/json"
        )
    
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
.9
