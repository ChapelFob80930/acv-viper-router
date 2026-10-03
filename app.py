"""FastAPI server: serves the demo UI and the decision API.   Run:  uvicorn app:app --port 8000"""
import json
import os
import threading
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core import model as M, pipeline
from core.demo_vehicles import VEHICLES
from core.exports import DESTINATIONS
from core.sim import TIER_MONTHS

ROOT = Path(__file__).parent
app = FastAPI(title="Viper Router demo")
model = M.load()
metrics = json.loads((M.MODEL_DIR / "metrics.json").read_text())


@app.get("/api/vehicles")
def vehicles():
    return VEHICLES


@app.get("/api/floorplan/demo")
def floorplan_demo():
    """Pre-computed synthetic floorplan examples (dealers, vehicles, transactions). Rebuild with build_floorplan.py."""
    return json.loads((ROOT / "data" / "floorplan_demo.json").read_text())


@app.get("/api/metrics")
def get_metrics():
    return {"metrics": metrics, "tier_months": TIER_MONTHS, "destinations": list(DESTINATIONS)}


@app.post("/api/decide")
def decide(v: dict):
    return pipeline.decide(v, model)


class Review(BaseModel):
    vin: str
    chosen_route: str
    model_route: str
    nets: dict
    note: str = ""


@app.post("/api/review")
def review(r: Review):
    return pipeline.log_specialist_review(r.vin, r.chosen_route, r.model_route, r.nets, r.note)


@app.post("/api/retrain")
def retrain():
    """Runs the automated learning loop now and hot-swaps the model if the candidate wins."""
    global model
    import retrain as R
    rec = R.run_once()
    if rec["promoted"]:
        model = M.load()
    return rec


def _auto_loop(minutes):
    while True:
        time.sleep(minutes * 60)
        try:
            retrain()
        except Exception as e:  # never kill the server on a failed retrain
            print("auto-retrain failed:", e)


if os.environ.get("VIPER_AUTO_RETRAIN_MIN"):
    threading.Thread(target=_auto_loop, args=(float(os.environ["VIPER_AUTO_RETRAIN_MIN"]),), daemon=True).start()

app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")
