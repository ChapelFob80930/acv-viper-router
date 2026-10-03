"""Synthetic market simulator.

ALL numbers here are illustrative placeholders. In production this module is replaced by
ACV's historical auction outcomes (sale price, sold/no-sale, fees) joined to Viper scans
and service history. The simulator exists so the pipeline, the model and the demo run end-to-end.
"""
import numpy as np
import pandas as pd

SEGMENTS = {  # segment -> typical new price
    "sedan": 26000, "suv": 33000, "truck": 42000, "hatchback": 23000, "luxury": 55000,
}
SEG_CODE = {s: i for i, s in enumerate(SEGMENTS)}

FEATURES = [
    "age", "mileage", "segment_code", "base_value",
    # Viper scan
    "dent_count", "damage_severity", "broken_parts", "brake_pad_mm",
    "oil_degradation", "engine_score", "est_repair_cost",
    # service history
    "avg_oil_interval", "longest_gap_months", "repeat_repairs", "open_recalls",
    "collision_events", "history_months", "service_tier",
    # market
    "local_demand", "export_score",
]
GATES = ["frame_damage", "flood", "salvage_title", "airbag_deployed", "odometer_mismatch"]
ROUTES = ["retain", "acv", "copart"]
ROUTE_LABEL = {"retain": "Retain & service", "acv": "ACV wholesale auction", "copart": "Copart"}

TIER_MONTHS = {1: 12, 2: 36, 3: 72, 4: 120}  # PLACEHOLDER tier definitions (not yet defined by the business)

ACV_FEE, ACV_RELIST = 450.0, 200.0
COPART_FEE = 500.0
RETAIL_RECON, RETAIL_HOLD_PER_DAY = 1900.0, 44.0


def est_repair_cost(dent_count, broken_parts, damage_severity, brake_pad_mm, oil_degradation, engine_score):
    """Deterministic repair estimate from scan readings (what Viper would output)."""
    brake = np.where(brake_pad_mm < 3, 480.0, np.where(brake_pad_mm < 5, 230.0, 0.0))
    return (140.0 * dent_count + 450.0 * broken_parts + 160.0 * damage_severity + brake
            + 350.0 * oil_degradation + 45.0 * np.clip(70 - engine_score, 0, None))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def expected_economics(d: pd.DataFrame) -> pd.DataFrame:
    """Noise-free expected net value of each route + P(sell at ACV). Uses gate flags if present."""
    bv = d["base_value"].to_numpy(float)
    rep = d["est_repair_cost"].to_numpy(float)
    coll = d["collision_events"].to_numpy(float)
    rec = d["open_recalls"].to_numpy(float)
    rr = d["repeat_repairs"].fillna(0.6).to_numpy(float)
    oi = d["avg_oil_interval"].fillna(7000).to_numpy(float)
    gap = d["longest_gap_months"].fillna(8).to_numpy(float)
    hist = d["history_months"].to_numpy(float)
    dem = d["local_demand"].to_numpy(float)
    exp = d["export_score"].to_numpy(float)

    gate_pen = np.zeros(len(d))
    for g, w in [("frame_damage", .60), ("flood", .70), ("salvage_title", .50), ("airbag_deployed", .42), ("odometer_mismatch", .65)]:
        if g in d:
            gate_pen += w * d[g].to_numpy(float)

    # --- ACV wholesale auction
    f = 0.93 - 1.35 * rep / bv - 0.045 * coll - 0.015 * rec - 0.035 * rr
    f += np.where(oi <= 6500, 0.05, np.where(oi > 9500, -0.07, 0.0))
    f -= 0.007 * np.clip(gap - 9, 0, 30)
    f += 0.0004 * hist
    f -= gate_pen
    f = np.clip(f, 0.10, 1.05)
    p_sell = _sigmoid(4.0 * (f - 0.55) + 1.5 * (dem - 1.0))
    acv_price = bv * f * dem ** 0.5
    net_acv = p_sell * (acv_price - ACV_FEE) - (1 - p_sell) * ACV_RELIST

    # --- Retain & service (retail through the dealer's own lot)
    fr = 0.985 - 0.05 * coll - 0.01 * rec - 0.015 * rr - 0.004 * np.clip(gap - 9, 0, 30) - gate_pen
    fr = np.clip(fr, 0.1, 1.0)
    retail_price = bv * 1.05 * fr * dem ** 2.0
    p_retail = _sigmoid(5.0 * (dem - 0.85) + 0.5)
    days = 55 + 25 * np.clip(1 - dem, 0, 1)
    cost = rep * 1.15 + RETAIL_RECON + RETAIL_HOLD_PER_DAY * days
    net_retain = p_retail * retail_price + (1 - p_retail) * 0.88 * bv * fr - cost

    # --- Copart (salvage / export)
    salvage_price = bv * (0.20 + 0.10 * exp)
    net_copart = 0.96 * salvage_price - COPART_FEE

    return pd.DataFrame({"exp_net_retain": net_retain, "exp_net_acv": net_acv, "exp_net_copart": net_copart,
                         "exp_p_sell_acv": p_sell}, index=d.index)


def generate(n=30000, seed=7, demand_shift=1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    segs = list(SEGMENTS)
    seg = rng.choice(segs, n, p=[.30, .28, .17, .15, .10])
    age = rng.integers(1, 13, n)
    mileage = np.clip(age * rng.normal(12000, 3500, n) + rng.normal(0, 4000, n), 3000, 260000)
    msrp = np.array([SEGMENTS[s] for s in seg]) * rng.normal(1, 0.08, n)
    base_value = np.clip(msrp * np.exp(-0.085 * age) * np.exp(-0.0000035 * mileage), 1800, None)

    latent = rng.beta(1.2, 6, n)  # hidden "how rough has this car been treated"
    dent_count = rng.poisson(latent * 14)
    damage_severity = np.clip(latent * 10 + rng.normal(0, .8, n), 0, 10)
    broken_parts = rng.poisson(latent * 3.5)
    brake_pad_mm = np.clip(11 - mileage / 18000 + rng.normal(0, 1.6, n), 1, 12)

    tier = rng.integers(1, 5, n)
    hist_m = np.minimum(np.array([TIER_MONTHS[t] for t in tier]), age * 12)
    avg_oil = np.clip(rng.normal(6500, 2200, n), 3000, 16000)
    longest_gap = np.clip(rng.exponential(6, n) + 1, 1, 48)
    repeat_rep = rng.poisson(0.35 + 0.08 * age)
    open_rec = rng.poisson(0.4, n)
    collisions = rng.poisson(0.15 + 1.6 * latent)
    # thinner history tiers leave gaps in the maintenance features -> NaN (LightGBM handles natively)
    repeat_rep = repeat_rep.astype(float)
    # (true maintenance values drive the economics; thin service tiers hide them from the model -> NaN, applied below)

    oil_deg = np.clip((np.nan_to_num(avg_oil, nan=7000) - 5000) / 12000 + rng.normal(0, .25, n), 0, 1)
    engine = np.clip(95 - 25 * oil_deg - 1.5 * age + rng.normal(0, 6, n), 20, 100)

    export_score = np.where(rng.random(n) < 0.3, 0.0, rng.uniform(0.4, 1.5, n))
    local_demand = np.clip(rng.normal(1.0 * demand_shift, 0.18, n), 0.55, 1.5)

    df = pd.DataFrame({
        "age": age, "mileage": mileage, "segment_code": [SEG_CODE[s] for s in seg], "base_value": base_value,
        "dent_count": dent_count, "damage_severity": damage_severity, "broken_parts": broken_parts,
        "brake_pad_mm": brake_pad_mm, "oil_degradation": oil_deg, "engine_score": engine,
        "avg_oil_interval": avg_oil, "longest_gap_months": longest_gap, "repeat_repairs": repeat_rep,
        "open_recalls": open_rec, "collision_events": collisions, "history_months": hist_m,
        "service_tier": tier, "local_demand": local_demand, "export_score": export_score,
    })
    df["est_repair_cost"] = est_repair_cost(df.dent_count, df.broken_parts, df.damage_severity,
                                            df.brake_pad_mm, df.oil_degradation, df.engine_score)
    # hard-gate flags (correlated with the latent roughness)
    df["frame_damage"] = rng.random(n) < 0.01 + 0.30 * latent ** 2
    df["flood"] = rng.random(n) < 0.012
    df["salvage_title"] = rng.random(n) < 0.03
    df["airbag_deployed"] = rng.random(n) < 0.01 + 0.15 * latent ** 2
    df["odometer_mismatch"] = rng.random(n) < 0.012

    # ---- realized outcomes (what the model is trained on): noisy versions of the expected economics
    ex = expected_economics(df)
    df = pd.concat([df, ex], axis=1)
    sold = rng.random(n) < df.exp_p_sell_acv
    df["sold_acv"] = sold.astype(int)
    px_noise = lambda s: np.exp(rng.normal(0, s, n))
    acv_price = (df.exp_net_acv - (1 - df.exp_p_sell_acv) * -ACV_RELIST) / np.maximum(df.exp_p_sell_acv, 1e-3) + ACV_FEE
    df["net_acv"] = np.where(sold, acv_price * px_noise(.07) - ACV_FEE, -ACV_RELIST)
    df["net_retain"] = df.exp_net_retain + rng.normal(0, 0.05, n) * df.base_value + df.est_repair_cost * rng.normal(0, .12, n)
    df["net_copart"] = (df.exp_net_copart + COPART_FEE) * px_noise(.12) - COPART_FEE
    df["gate_fail"] = df[GATES].any(axis=1)
    # thinner service packages leave maintenance features unobserved (LightGBM handles NaN natively)
    p_hidden = pd.Series(tier).map({1: .75, 2: .40, 3: .12, 4: 0.0}).to_numpy()
    hide = rng.random(n) < p_hidden
    for c in ("avg_oil_interval", "longest_gap_months", "repeat_repairs"):
        df.loc[hide, c] = np.nan
    return df



