"""Curated dealers/vehicles for the floorplan demo + helpers that evaluate them end to end."""
import json
from pathlib import Path

import pandas as pd

from . import floorplan as F

HERE = Path(__file__).parent
CURATED_DEALERS = list(json.loads((HERE / "curated_dealers.json").read_text()).values())

V = lambda **k: dict(dict(stolen_flag=False, flood=False, ca_admissible=True, mx_class_ok=True, open_recall=False,
                          service_score=.8, demand_index=1.0, damage_severity=1.0, title_status="clean"), **k)
CURATED_VEHICLES = [
    V(vehicle_id="VEHICLE_001", label="2018 Ford F-150", segment="truck", model_year=2018, mileage=98000, acv_price_cad=18500, demand_index=1.06),
    V(vehicle_id="VEHICLE_002", label="2023 Honda Civic", segment="hatchback", model_year=2023, mileage=30000, acv_price_cad=24500),
    V(vehicle_id="VEHICLE_003", label="2017 Toyota Tacoma", segment="truck", model_year=2017, mileage=112000, acv_price_cad=13200, title_status="rebuilt", damage_severity=3.0),
    V(vehicle_id="VEHICLE_004", label="2019 BMW X5", segment="luxury", model_year=2019, mileage=76000, acv_price_cad=33500, ca_admissible=False),
    V(vehicle_id="VEHICLE_005", label="2020 Ford Explorer", segment="suv", model_year=2020, mileage=64000, acv_price_cad=35000, demand_index=.5),
    V(vehicle_id="VEHICLE_006", label="2018 Jeep Grand Cherokee", segment="suv", model_year=2018, mileage=88000, acv_price_cad=19800, stolen_flag=True),
    # the frame-damaged F-150 from the routing demo (branded title)
    V(vehicle_id="VEHICLE_007", label="2017 Ford F-150 (frame damage)", segment="truck", model_year=2017, mileage=118000, acv_price_cad=9780,
      damage_severity=7.5, title_status="rebuilt", service_score=.6),
]


def prep_vehicle(v: dict) -> pd.Series:
    d = dict(v)
    d["age"] = F.CURRENT_YEAR - d["model_year"]
    d["segment_code"] = F.SEGMENTS.index(d["segment"])
    d["title_code"] = ["clean", "rebuilt", "salvage"].index(d["title_status"])
    return pd.Series(d)


def evaluate_all(dm, vm):
    dealers = {d["dealer_id"]: F.evaluate_dealer(pd.Series(d), dm) for d in CURATED_DEALERS}
    vehicles = {}
    for v in CURATED_VEHICLES:
        s = prep_vehicle(v)
        per = {}
        for c in ("Canada", "Mexico"):
            el = F.import_eligibility(s, c)
            ec = F.vehicle_economics(s, c, vm)
            if el["status"] == "ELIGIBLE" and ec["econ_status"].startswith("ELIGIBLE BUT"):
                el = dict(el, status="ELIGIBLE BUT NOT ECONOMIC", reasons=["Landed cost exceeds predicted resale value"])
            per[c] = {"eligibility": el, "economics": ec}
        vehicles[v["vehicle_id"]] = {"label": v["label"], "title": v["title_status"], "countries": per}
    return dealers, vehicles


def transaction(dealer_eval: dict, vehicle_eval: dict, country: str) -> dict:
    c = vehicle_eval["countries"][country]
    veh = dict(c["economics"], title=vehicle_eval["title"])
    return F.transaction_decision(dealer_eval, veh, c["eligibility"])
