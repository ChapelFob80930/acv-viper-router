"""Build the synthetic floorplan datasets and the demo JSON the 3D scene / API use."""
import json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from core import floorplan as F, floorplan_demo as D

if __name__ == "__main__":
    dm, vm = F.train_dealer_model(), F.train_vehicle_model()
    dealers, vehicles = D.evaluate_all(dm, vm)
    # ---- datasets (curated rows first)
    pf = F.gen_dealers(800 - len(D.CURATED_DEALERS), seed=2026)
    pf["pd_model"] = dm.predict_pd(pf)
    pf["risk_band"] = pf.pd_model.map(F.risk_band)
    pf["recommended_credit_line"] = [F.recommended_credit_line(r, r.pd_model)["line"] for _, r in pf.iterrows()]
    cur = pd.DataFrame(D.CURATED_DEALERS)
    cur["pd_model"] = [dealers[i]["pd"] for i in cur.dealer_id]; cur["risk_band"] = cur.pd_model.map(F.risk_band)
    cur["recommended_credit_line"] = [dealers[i]["credit_line"] for i in cur.dealer_id]
    alld = pd.concat([cur, pf], ignore_index=True)
    alld.to_csv("data/dealers_synthetic.csv", index=False)
    pv = F.gen_vehicles(1000 - len(D.CURATED_VEHICLES), seed=2026)
    rows = []
    for _, r in pv.iterrows():
        c = r.country
        el = F.import_eligibility(r, c); ec = F.vehicle_economics(r, c, vm)
        st = "ELIGIBLE BUT NOT ECONOMIC" if el["status"] == "ELIGIBLE" and ec["margin"] <= 0 else el["status"]
        rows.append(dict(vehicle_id=r.vehicle_id, country=c, segment=r.segment, model_year=r.model_year, mileage=round(r.mileage), title_status=r.title_status,
                         acv_price_cad=round(r.acv_price_cad), landed_cost=round(ec["landed"]), predicted_resale=round(ec["resale"]), expected_margin=round(ec["margin"]),
                         expected_days_to_sell=round(ec["days"]), recovery_value=round(ec["recovery"]), vehicle_risk=ec["band"], import_status=st))
    cv = []
    for vid, v in vehicles.items():
        for c, x in v["countries"].items():
            ec = x["economics"]
            cv.append(dict(vehicle_id=vid, country=c, label=v["label"], title_status=v["title"], import_status=x["eligibility"]["status"],
                           reason="; ".join(x["eligibility"]["reasons"]), acv_price_cad=round(ec["price"]), landed_cost=round(ec["landed"]),
                           predicted_resale=round(ec["resale"]), expected_margin=round(ec["margin"]), expected_days_to_sell=round(ec["days"]),
                           recovery_value=round(ec["recovery"]), vehicle_risk=ec["band"]))
    pd.concat([pd.DataFrame(cv), pd.DataFrame(rows)], ignore_index=True).to_csv("data/vehicles_synthetic.csv", index=False)
    # ---- demo JSON
    tx = {}
    for key, (vid, did, c) in {"approve": ("VEHICLE_001", "DEALER_001", "Canada"), "branded": ("VEHICLE_007", "DEALER_001", "Canada"),
                               "utilization": ("VEHICLE_001", "DEALER_004", "Canada"), "not_economic": ("VEHICLE_005", "DEALER_001", "Canada"),
                               "ineligible": ("VEHICLE_002", "DEALER_005", "Mexico"), "weak_dealer": ("VEHICLE_001", "DEALER_003", "Canada")}.items():
        tx[key] = dict(vehicle=vid, dealer=did, country=c, **D.transaction(dealers[did], vehicles[vid], c))
    out = dict(dealers=dealers, vehicles=vehicles, transactions=tx,
               portfolio=dict(n=len(alld), band_counts=alld.risk_band.value_counts().to_dict(), mean_pd=float(alld.pd_model.mean())))
    json.dump(out, open("data/floorplan_demo.json", "w"), indent=1, default=float)
    print("dealers", len(alld), alld.risk_band.value_counts().to_dict(), "vehicles", len(pv) + len(D.CURATED_VEHICLES) * 2)
    for k, t in tx.items():
        print(k, t["decision"], t["reasons"][:3], round(t["acv_funds"]), round(t["dealer_contributes"]), round(t["ltv"] * 100, 1))
    for vid in ("VEHICLE_001", "VEHICLE_005", "VEHICLE_007"):
        ec = vehicles[vid]["countries"]["Canada"]
        print(vid, ec["eligibility"]["status"], {k: round(ec["economics"][k]) for k in ("price", "transport", "import_total", "landed", "resale", "margin", "days", "recovery")}, ec["economics"]["band"])
