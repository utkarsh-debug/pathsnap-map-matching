"""PathSnap FastAPI backend (Laptop 1).
Run:  uvicorn api.main:app --host 0.0.0.0 --port 8000
Endpoints: GET /health, GET /trips, POST /predict {"trip_id": "..."}
Every request is logged to the console (shown to the evaluator during the demo).
"""
import logging
import re
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .pipeline import Pipeline
from .scoring import score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("pathsnap.api")
TRIP_ID = re.compile(r"^[0-9A-Za-z_-]{1,64}$")          # also blocks path traversal
STATE = {}


@asynccontextmanager
async def lifespan(app):
    STATE["pipeline"] = Pipeline()                       # model + decoder are loaded once, at start-up
    log.info("trips served: %s", [t["trip_id"] for t in STATE["pipeline"]._trip_cache()])
    yield


app = FastAPI(title="PathSnap API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.perf_counter()
    client = request.client.host if request.client else "?"
    log.info(">>> %s %s from %s", request.method, request.url.path, client)
    resp = await call_next(request)
    log.info("<<< %s %s -> %s in %.0f ms", request.method, request.url.path, resp.status_code,
             (time.perf_counter() - t0) * 1000)
    return resp


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    p = STATE["pipeline"]
    return {"status": "ok", "model": p.cfg["model_name"],
            "decoder": "hmm_viterbi" if p.hmm else "classifier_only"}


@app.get("/trips")
def trips():
    return {"trips": STATE["pipeline"]._trip_cache()}


@app.post("/predict")
def predict(req: PredictRequest):
    p = STATE["pipeline"]
    if not TRIP_ID.match(req.trip_id) or not p.trip_exists(req.trip_id):
        log.warning("unknown trip_id %r", req.trip_id)
        raise HTTPException(status_code=404, detail="unknown trip_id")
    public, internals = p.run(req.trip_id)               # prediction: no reference data involved
    metrics = score(req.trip_id, internals)              # scoring: reference data, server-side only
    public["metrics"] = metrics
    log.info("trip %s: %d points | decoder=%s | %s | pipeline %.0f ms", req.trip_id, public["n_points"],
             public["decoder"], "no reference available" if metrics is None else
             f"Top-1 {metrics['top1_accuracy']:.4f} (classifier-only {metrics['top1_accuracy_classifier_only']:.4f}), "
             f"continuity {metrics['route_continuity']}, candidate miss {metrics['candidate_miss_rate']}",
             public["timings_ms"]["total_ms"])
    return public
