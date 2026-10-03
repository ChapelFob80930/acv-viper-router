"""ACV International Floorplan Intelligence (Canada / Mexico dealer floorplan).

ALL DATA IS SYNTHETIC. Rules, thresholds, uplifts, FX and costs are illustrative placeholders.
  Rules engine : import eligibility, advance rates, LTV, hard limits       (deterministic)
  ML           : dealer PD (LightGBM + SHAP), vehicle resale / days-to-sell / recovery (LightGBM)
  LLM          : document extraction + summaries only (see core/llm.py); never decides credit
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

CURRENT_YEAR = 2026
FX_USD_CAD = 1.37
MAX_ADVANCE = {"clean": 0.90, "rebuilt": 0.70, "salvage": 0.60}
PD_REVIEW, PD_DECLINE = 0.10, 0.15            # transaction level
UTIL_REVIEW = 0.85
MARGIN_REVIEW_PCT = 0.08
DAYS_REVIEW = 60

DEALER_FEATURES = [
    "years_in_business", "annual_revenue", "gross_margin", "ebitda_margin", "bank_balance_pct", "cash_flow_volatility",
    "debt_to_revenue", "monthly_vehicle_sales", "inventory_turnover_days", "aged_inventory_ratio",
    "prior_delinquency_count", "late_payment_count", "returned_payment_count", "acv_purchase_count",
    "acv_repayment_score", "average_days_to_repay", "historical_utilization", "current_utilization", "owner_guarantor_score",
]
DEALER_LABELS = {
    "years_in_business": "Years in business", "annual_revenue": "Annual revenue", "gross_margin": "Gross margin",
    "ebitda_margin": "EBITDA margin", "bank_balance_pct": "Average bank balance", "cash_flow_volatility": "Cash-flow volatility",
    "debt_to_revenue": "Existing debt vs revenue", "monthly_vehicle_sales": "Monthly vehicle sales",
    "inventory_turnover_days": "Inventory turnover (days)", "aged_inventory_ratio": "Aged inventory share",
    "prior_delinquency_count": "Prior delinquencies", "late_payment_count": "Late payments",
    "returned_payment_count": "Returned payments", "acv_purchase_count": "ACV purchase history",
    "acv_repayment_score": "ACV repayment history", "average_days_to_repay": "Average days to repay",
    "historical_utilization": "Historical utilization", "current_utilization": "Credit-line utilization",
    "owner_guarantor_score": "Owner / guarantor score",
}
SEGMENTS = ["sedan", "suv", "truck", "hatchback", "luxury"]
UPLIFT = {"sedan": 1.30, "suv": 1.38, "truck": 1.48, "hatchback": 1.28, "luxury": 1.32}   # local retail vs U.S. clean value
TITLE_FACTOR = {"clean": 1.0, "rebuilt": 0.72, "salvage": 0.58}

# ------------------------------------------------------------------ dealers
def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def pd_logit(d: pd.DataFrame) -> np.ndarray:
    return (-5.6 + 3.2 * d.debt_to_revenue + 1.8 * d.cash_flow_volatility + 0.035 * (d.inventory_turnover_days - 45)
            + 2.2 * d.aged_inventory_ratio + 0.55 * d.prior_delinquency_count + 0.14 * d.late_payment_count
            + 0.3 * d.returned_payment_count + 1.6 * (d.current_utilization > 0.85) - 2.4 * (d.acv_repayment_score - 0.8)
            - 0.035 * np.minimum(d.years_in_business, 20) - 9.0 * (d.bank_balance_pct - 0.03) - 0.004 * (d.owner_guarantor_score - 650)
            - 0.25 * np.log(d.monthly_vehicle_sales / 30))


def gen_dealers(n=800, seed=11, label_noise=True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    q = rng.normal(0, 1, n)
    country = rng.choice(["Canada", "Mexico"], n, p=[.6, .4])
    units = np.clip(rng.lognormal(3.45 + .12 * q, .45), 8, 140).round()
    unit_rev = np.clip(rng.normal(8300, 1500, n), 4500, 14000)
    rev = units * 12 * unit_rev * rng.normal(1, .05, n)
    d = pd.DataFrame({
        "dealer_id": [f"DEALER_{i + 1:04d}" for i in range(n)], "country": country,
        "years_in_business": np.clip(rng.lognormal(1.8 + .25 * q, .5), 1, 35).round(),
        "annual_revenue": rev, "monthly_vehicle_sales": units,
        "gross_margin": np.clip(.12 + .015 * q + rng.normal(0, .02, n), .05, .25),
        "ebitda_margin": np.clip(.075 + .012 * q + rng.normal(0, .012, n), .01, .16),
        "bank_balance_pct": np.clip(.036 + .01 * q + rng.normal(0, .008, n), .003, .12),
        "cash_flow_volatility": np.clip(.3 - .07 * q + rng.normal(0, .06, n), .08, .8),
        "debt_to_revenue": np.clip(.15 - .045 * q + rng.normal(0, .035, n), .01, .6),
        "inventory_turnover_days": np.clip(52 - 7 * q + rng.normal(0, 7, n), 20, 130),
        "aged_inventory_ratio": np.clip(.2 - .05 * q + rng.normal(0, .04, n), 0, .8),
        "prior_delinquency_count": rng.poisson(np.clip(.35 - .3 * q, .02, 3)),
        "late_payment_count": rng.poisson(np.clip(1.4 - .7 * q, .1, 8)),
        "returned_payment_count": rng.poisson(np.clip(.25 - .15 * q, .02, 2)),
        "acv_repayment_score": np.clip(.8 + .07 * q + rng.normal(0, .05, n), .2, 1.0),
        "average_days_to_repay": np.clip(30 - 4 * q + rng.normal(0, 4, n), 8, 90),
        "historical_utilization": np.clip(rng.beta(4, 3, n) + .02 * q, .05, .98),
        "current_utilization": np.clip(rng.beta(4, 3, n), .02, .99),
        "owner_guarantor_score": np.clip(650 + 55 * q + rng.normal(0, 30, n), 450, 840).round(),
        "fraud_flag": rng.random(n) < .012,
    })
    d["acv_purchase_count"] = rng.poisson(units * 1.8 * np.clip(1 + .1 * q, .3, 2)).astype(float)
    p = _sigmoid(pd_logit(d) + (3.0 * d.fraud_flag) + (rng.normal(0, .35, n) if label_noise else 0))
    d["default_within_90d"] = (rng.random(n) < p).astype(int)
    d["annual_revenue"] = d.annual_revenue.round(-3)
    return d


class DealerModel:
    def __init__(self, clf):
        self.clf = clf
        self.expl = shap.TreeExplainer(clf)

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(X[DEALER_FEATURES])[:, 1]

    def factors(self, row: pd.DataFrame, k=3):
        sv = np.asarray(self.expl.shap_values(row[DEALER_FEATURES]))
        sv = sv[1] if sv.ndim == 3 else sv
        sv = np.ravel(sv[0] if sv.ndim > 1 else sv)
        items = [(DEALER_FEATURES[i], float(sv[i])) for i in range(len(sv))]
        pos = sorted([x for x in items if x[1] < 0], key=lambda x: x[1])[:k]   # negative SHAP on PD = lowers risk
        neg = sorted([x for x in items if x[1] > 0], key=lambda x: -x[1])[:k]
        fmt = lambda t: [{"feature": f, "label": DEALER_LABELS[f], "shap": s} for f, s in t]
        return fmt(pos), fmt(neg)


def train_dealer_model(seed=3) -> DealerModel:
    tr = gen_dealers(7000, seed=seed)
    clf = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.04, num_leaves=15, min_child_samples=40, subsample=.8,
                             subsample_freq=1, colsample_bytree=.9, verbose=-1, random_state=0).fit(tr[DEALER_FEATURES], tr.default_within_90d)
    return DealerModel(clf)


CREDIT_K = 1.40


def recommended_credit_line(dealer: pd.Series, pd_: float) -> dict:
    """Rules layer on top of PD: capped by inventory need, cash-flow capacity, repayment history, debt and PD."""
    units, rev = dealer.monthly_vehicle_sales, dealer.annual_revenue
    unit_cost = rev / (units * 12) * (1 - dealer.gross_margin)
    inv_need = units * (dealer.inventory_turnover_days / 30) * unit_cost
    cap_inventory = CREDIT_K * inv_need
    cap_cash = 1.9 * dealer.ebitda_margin * rev + 1.2 * dealer.bank_balance_pct * rev
    debt_room = max(0.35, 1 - 0.9 * dealer.debt_to_revenue)
    mult = float(np.clip(1 - 3.5 * pd_, .2, 1)) * (.9 + .2 * dealer.acv_repayment_score) * debt_room ** .5
    line = min(cap_inventory * mult, cap_cash)
    line = 0 if pd_ > .20 or bool(dealer.get("fraud_flag", False)) else round(line / 10000) * 10000
    return {"line": float(line), "inventory_need": float(inv_need), "cap_inventory": float(cap_inventory), "cap_cash": float(cap_cash)}


def risk_band(pd_: float) -> str:
    return "Low" if pd_ < .05 else "Medium" if pd_ < .12 else "High"


def evaluate_dealer(d: pd.Series, model: DealerModel) -> dict:
    row = pd.DataFrame([d])
    p = float(model.predict_pd(row)[0])
    cl = recommended_credit_line(d, p)
    util_amt = float(d.current_credit_limit_used) if "current_credit_limit_used" in d else float(d.current_utilization * cl["line"])
    available = max(0.0, cl["line"] - util_amt)
    util_pct = util_amt / cl["line"] if cl["line"] else 1.0
    pos, neg = model.factors(row)
    if bool(d.get("fraud_flag", False)) or p > .20:
        status = "DECLINE"
    elif p > .10 or util_pct > UTIL_REVIEW or d.get("inconsistent_docs", False):
        status = "MANUAL REVIEW"
    else:
        status = "APPROVED"
    return {"dealer_id": d.get("dealer_id"), "name": d.get("name"), "country": d.get("country"), "pd": p, "band": risk_band(p),
            "credit_line": cl["line"], "utilization": util_amt, "utilization_pct": util_pct, "available": available,
            "status": status, "positive": pos, "negative": neg, "cap_detail": cl}


# ------------------------------------------------------------------ vehicles
VEH_FEATURES = ["age", "mileage", "segment_code", "acv_price_cad", "damage_severity", "title_code", "service_score", "landed_cost", "demand_index"]


def import_costs(price_cad: float, country: str, segment: str) -> dict:
    transport = 1200.0 if country == "Canada" else 1500.0
    duty = 0.0 if country == "Canada" else 0.10 * price_cad          # CUSMA-originating vehicles: no duty into Canada (illustrative)
    tax = (0.05 if country == "Canada" else 0.16) * (price_cad + transport + duty)
    compliance = 700.0 if country == "Canada" else 900.0
    return {"transport": transport, "duty": duty, "tax": tax, "compliance": compliance, "import_total": duty + tax + compliance}


def gen_vehicles(n=1000, seed=21) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    seg = rng.choice(SEGMENTS, n, p=[.28, .27, .2, .12, .13])
    age = rng.integers(2, 12, n)
    mileage = np.clip(age * rng.normal(14000, 4000, n), 8000, 240000)
    base = np.array({"sedan": 30000, "suv": 40000, "truck": 50000, "hatchback": 27000, "luxury": 68000}[s] for s in seg) if False else \
        np.array([{"sedan": 30000, "suv": 40000, "truck": 50000, "hatchback": 27000, "luxury": 68000}[s] for s in seg], float)
    clean_value = base * np.exp(-.09 * age) * np.exp(-.0000035 * mileage) * rng.normal(1, .06, n)
    sev = np.clip(rng.gamma(1.4, 1.2, n), 0, 10)
    title = rng.choice(["clean", "rebuilt", "salvage"], n, p=[.88, .08, .04])
    tf = np.array([TITLE_FACTOR[t] for t in title])
    acv_price = clean_value * tf * (1 - .04 * sev) * rng.normal(.98, .04, n)
    demand = np.clip(rng.normal(1, .2, n), .4, 1.5)
    country = rng.choice(["Canada", "Mexico"], n, p=[.6, .4])
    rows = [import_costs(p, c, s) for p, c, s in zip(acv_price, country, seg)]
    landed = acv_price + np.array([r["transport"] + r["import_total"] for r in rows])
    resale = clean_value * np.array([UPLIFT[s] for s in seg]) * tf * (1 - .03 * sev) * demand ** .6 * rng.normal(1, .04, n)
    days = np.clip(30 + 25 * (1 - demand) + 4 * sev + (title != "clean") * 14 + resale / 2500 + rng.normal(0, 6, n), 10, 160)
    recovery = resale * np.clip(.86 - .02 * sev - (title != "clean") * .07, .4, .92) * rng.normal(1, .03, n)
    v = pd.DataFrame({"vehicle_id": [f"VEHICLE_{i + 1:04d}" for i in range(n)], "segment": seg, "model_year": CURRENT_YEAR - age, "age": age,
                      "mileage": mileage, "segment_code": [SEGMENTS.index(s) for s in seg], "acv_price_cad": acv_price,
                      "damage_severity": sev, "title_status": title, "title_code": [["clean", "rebuilt", "salvage"].index(t) for t in title],
                      "service_score": np.clip(rng.normal(.75, .15, n), .2, 1), "landed_cost": landed, "demand_index": demand,
                      "country": country, "resale_true": resale, "days_true": days, "recovery_true": recovery})
    return v


class VehicleModel:
    def __init__(self, resale, days, recovery):
        self.m = {"resale": resale, "days": days, "recovery": recovery}

    def predict(self, row: pd.DataFrame) -> dict:
        X = row[VEH_FEATURES]
        return {k: float(m.predict(X)[0]) for k, m in self.m.items()}


def train_vehicle_model(seed=5) -> VehicleModel:
    tr = gen_vehicles(6000, seed=seed)
    P = dict(n_estimators=300, learning_rate=.05, num_leaves=31, min_child_samples=25, subsample=.8, subsample_freq=1, verbose=-1, random_state=0)
    f = lambda y: lgb.LGBMRegressor(**P).fit(tr[VEH_FEATURES], tr[y])
    return VehicleModel(f("resale_true"), f("days_true"), f("recovery_true"))


# ------------------------------------------------------------------ rules: import eligibility
def import_eligibility(v: pd.Series, country: str) -> dict:
    """Deterministic. ELIGIBLE / INELIGIBLE / MANUAL REVIEW (economics are checked separately)."""
    inel, rev = [], []
    if v.get("stolen_flag", False):
        inel.append("Stolen vehicle flag")
    if v.get("flood", False):
        inel.append("Flood damage")
    if country == "Canada":
        if v.get("ca_admissible", True) is False:
            inel.append("Compliance / admissibility (RIV)")
        if v.get("open_recall", False):
            rev.append("Open recall must be cleared")
    else:
        if v.age < 8:
            inel.append("Model year does not meet the configured import route")
        if not v.get("mx_class_ok", True):
            inel.append("Vehicle classification not importable")
    if v.title_status != "clean":
        rev.append(f"{'Rebuilt' if v.title_status == 'rebuilt' else 'Salvage'} / branded title")
    status = "INELIGIBLE" if inel else "MANUAL REVIEW" if rev else "ELIGIBLE"
    return {"country": country, "status": status, "reasons": inel or rev}


def vehicle_economics(v: pd.Series, country: str, vm: VehicleModel) -> dict:
    ic = import_costs(float(v.acv_price_cad), country, v.segment)
    landed = float(v.acv_price_cad) + ic["transport"] + ic["import_total"]
    row = pd.DataFrame([dict(v, landed_cost=landed)])
    pr = vm.predict(row)
    margin = pr["resale"] - landed
    status = "ECONOMICALLY ATTRACTIVE" if margin > 0 else "ELIGIBLE BUT NOT ECONOMIC"
    liq = float(np.clip(1.15 - pr["days"] / 90, 0, 1))
    band = "Low" if (margin / landed >= .15 and pr["days"] <= 45 and pr["recovery"] >= landed) else \
           "High" if (margin <= 0 or pr["recovery"] < .75 * landed or pr["days"] > 75) else "Medium"
    return {"price": float(v.acv_price_cad), **ic, "landed": landed, "resale": pr["resale"], "days": pr["days"], "recovery": pr["recovery"],
            "margin": margin, "margin_pct": margin / landed, "liquidity": liq, "band": band, "econ_status": status}


# ------------------------------------------------------------------ transaction engine (rules + ML outputs)
def transaction_decision(dealer: dict, veh: dict, elig: dict, fraud=False) -> dict:
    rs_dec, rs_rev, rs_ok = [], [], []
    cap = MAX_ADVANCE.get(veh.get("title", "clean"), .9)
    funding_needed = veh["landed"]
    max_funding = cap * veh["resale"]
    funding = min(funding_needed, max_funding)
    ltv = funding / veh["resale"]
    dealer_contrib = funding_needed - funding
    after = (dealer["utilization"] + funding) / dealer["credit_line"] if dealer["credit_line"] else 9.9
    # declines
    if elig["status"] == "INELIGIBLE": rs_dec.append("Import ineligible")
    if veh["margin"] <= 0: rs_dec.append("Vehicle economics negative")
    if dealer["available"] < funding: rs_dec.append("Dealer credit line exhausted")
    if dealer["pd"] > PD_DECLINE: rs_dec.append("Dealer PD above threshold")
    if fraud or dealer["status"] == "DECLINE": rs_dec.append("Fraud flag / dealer declined")
    # reviews
    if elig["status"] == "MANUAL REVIEW": rs_rev += elig["reasons"]
    if PD_REVIEW < dealer["pd"] <= PD_DECLINE: rs_rev.append("Dealer PD above preferred level")
    if after > UTIL_REVIEW and dealer["available"] >= funding: rs_rev.append(f"Dealer utilization above {int(UTIL_REVIEW * 100)}% after funding")
    if 0 < veh["margin_pct"] < MARGIN_REVIEW_PCT: rs_rev.append("Margin below preferred threshold")
    if veh["recovery"] < funding: rs_rev.append("Recovery value below exposure")
    if veh["days"] > DAYS_REVIEW: rs_rev.append("Expected time-to-sale above policy")
    # approvals (shown only on approve)
    if dealer["pd"] <= PD_REVIEW: rs_ok.append("Dealer PD within policy")
    if dealer["available"] >= funding: rs_ok.append("Available credit sufficient")
    if elig["status"] == "ELIGIBLE": rs_ok.append("Import eligibility passed")
    if veh["margin"] > 0: rs_ok.append("Vehicle margin positive")
    if ltv <= cap: rs_ok.append("LTV within policy")
    if veh["recovery"] >= funding: rs_ok.append("Recovery value acceptable")
    if veh["days"] <= DAYS_REVIEW: rs_ok.append("Expected time-to-sale acceptable")
    decision = "DECLINE" if rs_dec else "MANUAL REVIEW" if rs_rev else "APPROVE"
    return {"decision": decision, "reasons": rs_dec if rs_dec else rs_rev if rs_rev else rs_ok, "approve_reasons": rs_ok,
            "funding_needed": funding_needed, "acv_funds": funding, "dealer_contributes": dealer_contrib, "ltv": ltv, "max_advance": cap,
            "util_after": after}
