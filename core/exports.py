"""Copart workflow: export eligibility check (title, destination import rules, buyer demand).

DESTINATION RULES BELOW ARE ILLUSTRATIVE PLACEHOLDERS, not real import law.
"""

DESTINATIONS = {
    #  name          max_age  accepts_branded  accepts_flood  demand by segment (0-1.5)
    "Nigeria":      dict(max_age=15, branded=False, flood=False, demand=dict(sedan=1.2, suv=1.1, truck=0.9, hatchback=0.8, luxury=0.7), apr_tier="C"),
    "Ghana":        dict(max_age=10, branded=False, flood=False, demand=dict(sedan=1.1, suv=1.0, truck=0.8, hatchback=0.9, luxury=0.6), apr_tier="C"),
    "UAE":          dict(max_age=12, branded=True,  flood=False, demand=dict(sedan=0.7, suv=1.2, truck=1.3, hatchback=0.5, luxury=1.3), apr_tier="A"),
    "Mexico":       dict(max_age=12, branded=True,  flood=False, demand=dict(sedan=0.9, suv=1.0, truck=1.4, hatchback=0.8, luxury=0.6), apr_tier="B"),
    "Canada":       dict(max_age=15, branded=False, flood=False, demand=dict(sedan=0.9, suv=1.2, truck=1.3, hatchback=0.9, luxury=0.9), apr_tier="A"),
    "Jamaica":      dict(max_age=10, branded=True,  flood=False, demand=dict(sedan=1.0, suv=1.1, truck=0.9, hatchback=1.0, luxury=0.5), apr_tier="C"),
    "Dominican Rep.": dict(max_age=14, branded=True, flood=True, demand=dict(sedan=1.0, suv=1.2, truck=1.0, hatchback=0.9, luxury=0.6), apr_tier="B"),
}


def check_export(segment: str, age: int, gates: dict) -> dict:
    """Return per-destination eligibility and the best destination (highest buyer demand)."""
    branded = bool(gates.get("salvage_title") or gates.get("frame_damage") or gates.get("airbag_deployed"))
    flood = bool(gates.get("flood"))
    rows = []
    for name, r in DESTINATIONS.items():
        reasons = []
        if age > r["max_age"]:
            reasons.append(f"vehicle age {age} > import limit {r['max_age']} yrs")
        if branded and not r["branded"]:
            reasons.append("destination does not accept branded/salvage titles")
        if flood and not r["flood"]:
            reasons.append("destination bans flood-damaged vehicles")
        demand = r["demand"].get(segment, 0.7)
        rows.append({"destination": name, "eligible": not reasons, "demand": demand, "reasons": reasons, "apr_tier": r["apr_tier"]})
    ok = [r for r in rows if r["eligible"]]
    best = max(ok, key=lambda r: r["demand"]) if ok else None
    return {"destinations": rows, "best": best, "eligible": best is not None,
            "export_score": best["demand"] if best else 0.0,
            "path": "International auction" if best else "Domestic salvage auction"}
