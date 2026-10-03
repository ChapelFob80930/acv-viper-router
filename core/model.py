"""Layer 2: gradient-boosted route-value model + SHAP explanations."""
import json
import math
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from .sim import FEATURES, ROUTES

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

PRETTY = {
    "age": "Vehicle age", "mileage": "Mileage", "segment_code": "Body segment", "base_value": "Market base value",
    "dent_count": "Dent count (scan)", "damage_severity": "Damage severity (scan)", "broken_parts": "Broken parts (scan)",
    "brake_pad_mm": "Brake pad thickness (scan)", "oil_degradation": "Oil condition (scan)", "engine_score": "Engine score (scan)",
    "est_repair_cost": "Estimated repair cost", "avg_oil_interval": "Avg oil-change interval", "longest_gap_months": "Longest service gap",
    "repeat_repairs": "Repeat repairs, same part", "open_recalls": "Open recalls", "collision_events": "Collision / insurance events",
    "history_months": "Service history available", "service_tier": "Service package tier", "local_demand": "Local retail demand",
    "export_score": "Export buyer demand",
}

PARAMS = dict(n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=40, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.9, verbose=-1, random_state=0)


class RouteModel:
    def __init__(self, regs: dict, clf, qlo: dict, qhi: dict):
        self.regs, self.clf, self.qlo, self.qhi = regs, clf, qlo, qhi
        self._expl = {r: shap.TreeExplainer(m) for r, m in regs.items()}

    def predict(self, X: pd.DataFrame) -> dict:
        nets = {r: self.regs[r].predict(X) for r in ROUTES}
        lo = {r: self.qlo[r].predict(X) for r in ROUTES}
        hi = {r: self.qhi[r].predict(X) for r in ROUTES}
        return {"nets": nets, "lo": lo, "hi": hi, "p_sell_acv": self.clf.predict_proba(X)[:, 1]}

    @staticmethod
    def confidence(nets: dict, lo: dict, hi: dict) -> np.ndarray:
        """P(the top-ranked route is truly better than the runner-up), from the 10-90% prediction intervals."""
        n = np.column_stack([nets[r] for r in ROUTES])
        sig = np.maximum((np.column_stack([hi[r] for r in ROUTES]) - np.column_stack([lo[r] for r in ROUTES])) / 2.563, 50.0)
        order = np.argsort(-n, axis=1)
        i = np.arange(len(n))
        t, s2 = order[:, 0], order[:, 1]
        z = (n[i, t] - n[i, s2]) / np.sqrt(sig[i, t] ** 2 + sig[i, s2] ** 2)
        return 0.5 * (1 + np.vectorize(math.erf)(z / math.sqrt(2)))

    def explain(self, X: pd.DataFrame, route: str, top=6) -> list[dict]:
        sv = self._expl[route].shap_values(X)[0]
        order = np.argsort(-np.abs(sv))[:top]
        out = []
        for i in order:
            f = FEATURES[i]
            val = X.iloc[0, i]
            out.append({"feature": f, "label": PRETTY[f], "value": None if pd.isna(val) else float(val), "shap": float(sv[i])})
        return out

    def base_value(self, route: str) -> float:
        return float(np.ravel(self._expl[route].expected_value)[0])

    def payload(self) -> dict:
        return {"regs": self.regs, "clf": self.clf, "qlo": self.qlo, "qhi": self.qhi}


def train_models(df: pd.DataFrame):
    X = df[FEATURES]
    regs = {r: lgb.LGBMRegressor(**PARAMS).fit(X, df[f"net_{r}"]) for r in ROUTES}
    clf = lgb.LGBMClassifier(**{k: v for k, v in PARAMS.items()}).fit(X, df["sold_acv"])
    qp = dict(PARAMS, n_estimators=200)
    qlo = {r: lgb.LGBMRegressor(objective="quantile", alpha=0.10, **qp).fit(X, df[f"net_{r}"]) for r in ROUTES}
    qhi = {r: lgb.LGBMRegressor(objective="quantile", alpha=0.90, **qp).fit(X, df[f"net_{r}"]) for r in ROUTES}
    return RouteModel(regs, clf, qlo, qhi)


def save(model: RouteModel, metrics: dict):
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model.payload(), MODEL_DIR / "route_model.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))


def load() -> RouteModel:
    d = joblib.load(MODEL_DIR / "route_model.joblib")
    return RouteModel(d["regs"], d["clf"], d["qlo"], d["qhi"])
