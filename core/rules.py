"""Layer 1: deterministic rules engine. Hard gates run BEFORE the ML model."""

GATE_RULES = [
    ("frame_damage", "Frame / structural damage detected by Viper scan"),
    ("flood", "Flood damage indicator"),
    ("salvage_title", "Salvage or branded title"),
    ("airbag_deployed", "Airbag deployment on record"),
    ("odometer_mismatch", "Odometer mismatch between scan, service records and title"),
]


def run_gates(gates: dict) -> dict:
    """Return which gates failed. Any failure => skip the ML model and go to the salvage path."""
    results = [{"gate": g, "label": label, "failed": bool(gates.get(g, False))} for g, label in GATE_RULES]
    failed = [r for r in results if r["failed"]]
    return {
        "passed": not failed,
        "results": results,
        "failed": failed,
        "note": ("Frame-damaged cars can legally be resold in the US with disclosure/branded title, "
                 "but routing them directly avoids a failed auction cycle.") if any(r["gate"] == "frame_damage" for r in failed) else "",
    }
