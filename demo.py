"""CLI demo: run the 5 demo vehicles through the full pipeline."""
from core import model as M, pipeline
from core.demo_vehicles import VEHICLES

if __name__ == "__main__":
    mdl = M.load()
    for key, v in VEHICLES.items():
        r = pipeline.decide(v, mdl)
        m = r["steps"]["model"]
        print(f"\n=== {v['label']}: {r['title']} ===")
        print("events parsed:", len(r["steps"]["capture"]["events"]), f"({r['steps']['capture']['parse_mode']})")
        print("gates:", "PASS" if r["steps"]["gates"]["passed"] else "FAIL -> " + ", ".join(g["gate"] for g in r["steps"]["gates"]["failed"]))
        if m:
            print("nets:", {k: round(x) for k, x in m["nets"].items()}, f"margin {m['margin_pct']}%")
        print("ROUTE:", r["route_label"], "| review:", r["needs_review"])
        o = r["steps"]["outcome"]
        if o["kind"] == "copart":
            print("export:", o["export"]["path"], o["export"]["best"]["destination"] if o["export"]["best"] else "")
        print(r["explanation"]["text"])
