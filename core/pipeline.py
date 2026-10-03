"""Orchestrates the three layers for one vehicle."""
import json
import time
from pathlib import Path

import numpy as np

from . import exports, llm, rules
from .features import build_row, service_features
from .model import RouteModel
from .sim import ACV_FEE, COPART_FEE, ROUTE_LABEL, ROUTES

REVIEW_THRESHOLD = 0.10  # top-2 routes within 10% -> condition specialist decides
CONF_THRESHOLD = 0.55    # model confidence (P top route is truly best) below this -> human validation
REVIEW_LOG = Path(__file__).resolve().parent.parent / "models" / "specialist_reviews.jsonl"


def decide(v: dict, model: RouteModel) -> dict:
    t0 = time.time()
    out: dict = {"vin": v.get("vin"), "title": f"{v['year']} {v['make']} {v['model']}", "steps": {}}

    # ---- 1. DATA CAPTURE: Viper scan + service history parsed by the LLM
    parsed = llm.parse_service_records(v.get("service_text", ""))
    svc = service_features(parsed["events"], v["service_tier"], v["age"])
    out["steps"]["capture"] = {"scan": v["scan"], "events": parsed["events"], "parse_mode": parsed["mode"],
                               "service_features": {k: (None if (isinstance(x, float) and np.isnan(x)) else x) for k, x in svc.items()}}

    # ---- 2. LAYER 1: rules engine
    gates = rules.run_gates(v["gates"])
    out["steps"]["gates"] = gates
    exp = exports.check_export(v["segment"], v["age"], v["gates"])

    if not gates["passed"]:
        out.update(route="copart", route_label=ROUTE_LABEL["copart"], model_skipped=True, needs_review=False)
        out["steps"]["model"] = None
        out["steps"]["outcome"] = _copart_outcome(v, exp, price=None)
        ctx = {"decision": {"route": "copart", "route_label": ROUTE_LABEL["copart"]}, "gate_failed": gates["failed"], "export": exp}
        out["explanation"] = llm.explain_decision(ctx)
        out["latency_ms"] = round((time.time() - t0) * 1000)
        return out

    # ---- 3. LAYER 2: ML model
    X = build_row(v, svc, exp["export_score"])
    pred = model.predict(X)
    nets = {r: float(pred["nets"][r][0]) for r in ROUTES}
    ranked = sorted(nets, key=nets.get, reverse=True)
    top, second = ranked[0], ranked[1]
    margin = (nets[top] - nets[second]) / max(abs(nets[top]), 1.0)
    conf = float(model.confidence(pred["nets"], pred["lo"], pred["hi"])[0])
    low_margin, low_conf = margin < REVIEW_THRESHOLD, conf < CONF_THRESHOLD
    needs_review = low_margin or low_conf
    review_reason = ("low confidence and close margin" if low_margin and low_conf else
                     "low model confidence" if low_conf else "top two routes within 10%" if low_margin else None)
    drivers = model.explain(X, top)
    out["steps"]["model"] = {
        "nets": nets, "p_sell_acv": float(pred["p_sell_acv"][0]), "ranked": ranked,
        "margin_pct": round(margin * 100, 1), "threshold_pct": REVIEW_THRESHOLD * 100,
        "confidence_pct": round(conf * 100, 1), "confidence_threshold_pct": CONF_THRESHOLD * 100,
        "review_reason": review_reason,
        "intervals": {r: [float(pred["lo"][r][0]), float(pred["hi"][r][0])] for r in ROUTES},
        "drivers": drivers, "base_value": model.base_value(top),
        "features": {k: (None if np.isnan(x) else float(x)) for k, x in X.iloc[0].items()},
    }
    out.update(route=top, route_label=ROUTE_LABEL[top], model_skipped=False, needs_review=needs_review,
               runner_up=second, runner_up_label=ROUTE_LABEL[second])

    # ---- 4. LAYER 3: explanation (LLM explains, never decides)
    ctx = {"decision": {"route": top, "route_label": ROUTE_LABEL[top]}, "gate_failed": [], "nets": nets,
           "drivers": drivers, "needs_review": needs_review, "margin_pct": margin * 100, "confidence_pct": conf * 100, "review_reason": review_reason,
           "runner_up": second, "runner_up_label": ROUTE_LABEL[second],
           "vehicle": out["title"]}
    out["explanation"] = llm.explain_decision(ctx)

    # ---- 5. OUTCOME
    if top == "copart":
        price = v["base_value"] * (0.20 + 0.10 * exp["export_score"])
        out["steps"]["outcome"] = _copart_outcome(v, exp, price=price)
    elif top == "retain":
        out["steps"]["outcome"] = {"kind": "retain", "repair_estimate": float(X["est_repair_cost"].iloc[0]),
                                   "text": "Dealer reconditions and retails the car on their own lot."}
    else:
        out["steps"]["outcome"] = {"kind": "acv", "text": "Listed on ACV's wholesale auction.", "platform_fee": ACV_FEE,
                                   "p_sell": float(pred["p_sell_acv"][0])}
    out["latency_ms"] = round((time.time() - t0) * 1000)
    return out


def _copart_outcome(v, exp, price):
    price = price if price is not None else v["base_value"] * (0.20 + 0.10 * exp["export_score"])
    o = {"kind": "copart", "export": exp, "expected_price": round(price),
         "platform_fee": COPART_FEE,
         "family_note": "ACV and Copart are one group (merger): this is an internal hand-off, the fee stays in-house. No referral / rev-share needed."}
    o["floorplan_note"] = "Export-bound cars continue into dealer floorplan financing (ACV Capital): see the 3D journey and the floorplan flowchart."
    return o


def log_specialist_review(vin: str, chosen_route: str, model_route: str, nets: dict, note: str = "") -> dict:
    """Specialist overrides become training data (label = what a human expert decided)."""
    rec = {"ts": time.time(), "vin": vin, "specialist_route": chosen_route, "model_route": model_route,
           "agreed_with_model": chosen_route == model_route, "model_nets": nets, "note": note}
    REVIEW_LOG.parent.mkdir(exist_ok=True)
    with REVIEW_LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    n = sum(1 for _ in REVIEW_LOG.open())
    return {"logged": True, "total_labels": n}
