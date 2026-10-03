"""Generate synthetic data, train the route-value models, evaluate decision quality on a held-out fleet."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score

from core import model as M
from core.sim import FEATURES, ROUTES, generate

REVIEW_THRESHOLD = 0.10   # top-2 routes within 10% -> review
CONF_THRESHOLD = 0.55     # model confidence below this -> review (confidence = P(top route is truly best))
SPECIALIST_ACCURACY = 0.85  # ASSUMPTION: specialist picks the truly better route 85% of the time

if __name__ == "__main__":
    df = generate(40000, seed=7)
    train, test = df.iloc[:30000], df.iloc[30000:]
    tr_ok = train[~train.gate_fail]  # the model never sees gate-failed cars (they skip it)
    print(f"training on {len(tr_ok):,} gate-passing cars (of {len(train):,})")
    mdl = M.train_models(tr_ok)

    te_ok = test[~test.gate_fail]
    pred = mdl.predict(te_ok[FEATURES])
    metrics = {"n_train": len(tr_ok), "n_test": len(test), "regression": {}}
    for r in ROUTES:
        metrics["regression"][r] = {"r2": round(r2_score(te_ok[f"net_{r}"], pred["nets"][r]), 3),
                                    "mae": round(mean_absolute_error(te_ok[f"net_{r}"], pred["nets"][r]))}
    metrics["p_sell_auc"] = round(roc_auc_score(te_ok.sold_acv, pred["p_sell_acv"]), 3)

    # ---- decision-quality on the WHOLE held-out fleet (incl. gate-failed cars) using noise-free expected values
    exp_cols = [f"exp_net_{r}" for r in ROUTES]
    truth = test[exp_cols].to_numpy()
    nets = np.zeros((len(test), 3)); nets[:] = np.nan
    idx_ok = np.where(~test.gate_fail.to_numpy())[0]
    nets[idx_ok] = np.column_stack([pred["nets"][r] for r in ROUTES])
    gate_fail = test.gate_fail.to_numpy()

    s = np.sort(np.nan_to_num(nets, nan=0), axis=1)
    margin = (s[:, 2] - s[:, 1]) / np.maximum(np.abs(s[:, 2]), 1)
    conf = np.ones(len(test)); conf[idx_ok] = M.RouteModel.confidence(pred["nets"], pred["lo"], pred["hi"])
    review = (~gate_fail) & ((margin < REVIEW_THRESHOLD) | (conf < CONF_THRESHOLD))
    model_pick = np.where(gate_fail, 2, np.nanargmax(np.where(np.isnan(nets), -1e12, nets), axis=1))
    oracle_pick = truth.argmax(axis=1)
    rng = np.random.default_rng(1)
    spec_correct = rng.random(len(test)) < SPECIALIST_ACCURACY
    final_pick = np.where(review, np.where(spec_correct, oracle_pick, model_pick), model_pick)

    sweep = {}
    for th in (0.5, 0.6, 0.7, 0.75, 0.8, 0.9):
        rv = (~gate_fail) & ((margin < REVIEW_THRESHOLD) | (conf < th))
        auto = (~gate_fail) & ~rv
        sweep[str(th)] = {"review_rate_pct": round(100 * rv.mean(), 1),
                          "auto_agreement_pct": round(100 * (model_pick[auto] == oracle_pick[auto]).mean(), 1),
                          "reviewed_model_agreement_pct": round(100 * (model_pick[rv] == oracle_pick[rv]).mean(), 1) if rv.any() else None}
    val = lambda pick: truth[np.arange(len(test)), pick].mean()
    always_acv = val(np.full(len(test), 1))
    rules_only = val(np.where(gate_fail, 2, 1))
    system, model_only, oracle = val(final_pick), val(model_pick), val(oracle_pick)
    metrics["policy"] = {
        "avg_net_per_car": {"always_acv": round(always_acv), "rules_only": round(rules_only),
                            "model_no_human": round(model_only), "full_system": round(system), "oracle": round(oracle)},
        "uplift_vs_always_acv_per_car": round(system - always_acv),
        "uplift_vs_always_acv_pct": round(100 * (system / always_acv - 1), 1),
        "pct_of_oracle_captured": round(100 * system / oracle, 1),
        "agreement_with_oracle_pct": round(100 * (model_pick == oracle_pick).mean(), 1),
        "human_review_rate_pct": round(100 * review.mean(), 1),
        "gate_skip_rate_pct": round(100 * gate_fail.mean(), 1),
        "route_mix_pct": {k: round(100 * (final_pick == i).mean(), 1) for i, k in enumerate(ROUTES)},
        "specialist_accuracy_assumed": SPECIALIST_ACCURACY,
        "confidence_threshold": CONF_THRESHOLD, "margin_threshold": REVIEW_THRESHOLD,
        "threshold_sweep": sweep,
    }
    # uplift by service tier (does deeper history help?)
    by_tier = {}
    for t in (1, 2, 3, 4):
        m = (test.service_tier == t).to_numpy()
        by_tier[str(t)] = {"avg_net_system": round(truth[m, :][np.arange(m.sum()), final_pick[m]].mean()),
                           "avg_net_always_acv": round(truth[m, 1].mean()),
                           "pct_of_oracle": round(100 * truth[m, :][np.arange(m.sum()), final_pick[m]].mean() / truth[m].max(axis=1).mean(), 1),
                           "agreement_with_oracle_pct": round(100 * (model_pick[m] == oracle_pick[m]).mean(), 1)}
    metrics["by_tier"] = by_tier
    M.save(mdl, metrics)
    import json; print(json.dumps(metrics, indent=2))
