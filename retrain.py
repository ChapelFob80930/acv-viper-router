"""Automated learning loop: ingest new outcomes -> drift check -> retrain candidate -> promote only if better.
Run once:  python retrain.py --once      Run on a schedule:  python retrain.py --loop 60   (minutes)
Demo data: fresh 'outcomes' come from the simulator with a market shift (retail demand down 8%)."""
import argparse, json, time
import joblib, numpy as np
import pandas as pd
from core import model as M
from core.sim import FEATURES, ROUTES, generate

LOG = M.MODEL_DIR / "retrain_log.jsonl"


def psi(ref, new, bins=10):
    ref, new = ref.dropna(), new.dropna()
    if ref.nunique() < 3 or len(new) == 0:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(ref, edges)[0] / len(ref) + 1e-4
    b = np.histogram(new, edges)[0] / len(new) + 1e-4
    return float(((b - a) * np.log(b / a)).sum())


def policy_value(mdl, df):
    ok = df[~df.gate_fail]
    p = mdl.predict(ok[FEATURES])["nets"]
    pick = np.column_stack([p[r] for r in ROUTES]).argmax(axis=1)
    truth = ok[[f"exp_net_{r}" for r in ROUTES]].to_numpy()
    return float(truth[np.arange(len(ok)), pick].mean())


def run_once(shift=0.92):
    t = time.time()
    ref = generate(10000, seed=7)
    fresh = generate(12000, seed=int(t) % 100000, demand_shift=shift)
    drift = {f: round(psi(ref[f], fresh[f]), 3) for f in FEATURES}
    drifted = sorted([f for f, v in drift.items() if v > 0.2], key=drift.get, reverse=True)
    fresh_train, fresh_hold = fresh.iloc[:9000], fresh.iloc[9000:]
    labels = []
    rv = M.MODEL_DIR / "specialist_reviews.jsonl"
    n_labels = sum(1 for _ in rv.open()) if rv.exists() else 0
    current = M.load()
    cand_df = pd.concat([ref.iloc[:8000], fresh_train])
    cand = M.train_models(cand_df[~cand_df.gate_fail])
    cur_v, cand_v = policy_value(current, fresh_hold), policy_value(cand, fresh_hold)
    promoted = cand_v >= cur_v
    if promoted:
        joblib.dump(cand.payload(), M.MODEL_DIR / "route_model.joblib")
    rec = {"ts": t, "new_outcomes": len(fresh), "specialist_labels_seen": n_labels, "drifted_features": drifted,
           "max_psi": max(drift.values()), "current_avg_net": round(cur_v), "candidate_avg_net": round(cand_v), "promoted": promoted}
    with LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--once", action="store_true"); ap.add_argument("--loop", type=int, default=0)
    a = ap.parse_args()
    while True:
        print(json.dumps(run_once(), indent=2))
        if not a.loop:
            break
        time.sleep(a.loop * 60)
