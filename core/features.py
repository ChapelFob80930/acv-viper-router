"""Turn parsed service events + Viper scan + market info into the model's feature row."""
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .sim import FEATURES, SEG_CODE, TIER_MONTHS, est_repair_cost

DEMO_TODAY = date(2026, 10, 3)  # fixed so demo output is deterministic
NON_REPAIR = {"oil", "tires", "battery", "coolant"}


def service_features(events: list[dict], tier: int, age: int, today: date = DEMO_TODAY) -> dict:
    """Aggregate events inside the history window allowed by the car's service-package tier."""
    window_start = today - timedelta(days=30.4 * TIER_MONTHS[tier])
    ev = [e for e in events if e.get("date") and date.fromisoformat(e["date"]) >= window_start]
    ev.sort(key=lambda e: e["date"])

    oil = [e for e in ev if e["part"] == "oil" and e.get("mileage")]
    intervals = [b["mileage"] - a["mileage"] for a, b in zip(oil, oil[1:]) if b["mileage"] > a["mileage"]]
    avg_oil = float(np.mean(intervals)) if intervals else np.nan

    dates = [date.fromisoformat(e["date"]) for e in ev]
    if dates:
        pts = dates + [today]
        longest_gap = max((b - a).days for a, b in zip(pts, pts[1:])) / 30.4
        repairs = [e["part"] for e in ev if e["action"] in ("repair", "replace") and e["part"] not in NON_REPAIR]
        repeat = sum(1 for p in set(repairs) if repairs.count(p) >= 2)
    else:
        longest_gap, repeat = np.nan, np.nan
    hist_months = min(TIER_MONTHS[tier], age * 12)  # history the package makes visible
    return {"avg_oil_interval": avg_oil, "longest_gap_months": longest_gap,
            "repeat_repairs": float(repeat) if repeat == repeat else np.nan,
            "history_months": float(hist_months),
            "n_events_in_window": len(ev), "n_events_total": len(events), "window_months": TIER_MONTHS[tier]}


def build_row(v: dict, svc: dict, export_score: float) -> pd.DataFrame:
    s = v["scan"]
    row = {
        "age": v["age"], "mileage": v["mileage"], "segment_code": SEG_CODE[v["segment"]], "base_value": v["base_value"],
        "dent_count": s["dent_count"], "damage_severity": s["damage_severity"], "broken_parts": s["broken_parts"],
        "brake_pad_mm": s["brake_pad_mm"], "oil_degradation": s["oil_degradation"], "engine_score": s["engine_score"],
        "avg_oil_interval": svc["avg_oil_interval"], "longest_gap_months": svc["longest_gap_months"],
        "repeat_repairs": svc["repeat_repairs"], "open_recalls": v["open_recalls"],
        "collision_events": v["collision_events"], "history_months": svc["history_months"],
        "service_tier": v["service_tier"], "local_demand": v["local_demand"], "export_score": export_score,
    }
    row["est_repair_cost"] = float(est_repair_cost(row["dent_count"], row["broken_parts"], row["damage_severity"],
                                                   row["brake_pad_mm"], row["oil_degradation"], row["engine_score"]))
    return pd.DataFrame([row])[FEATURES]
