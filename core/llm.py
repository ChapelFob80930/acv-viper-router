"""Layer 3: LLM. (a) parse messy service records into structured events, (b) explain the decision.
The LLM NEVER makes the routing decision.

Uses the Claude API when ANTHROPIC_API_KEY is set; otherwise falls back to a deterministic
regex parser / template so the demo always runs offline. The response says which mode was used.
"""
import json
import os
import re
from datetime import date

PARSE_MODEL = os.environ.get("VIPER_PARSE_MODEL", "claude-haiku-4-5-20251001")
EXPLAIN_MODEL = os.environ.get("VIPER_EXPLAIN_MODEL", "claude-haiku-4-5-20251001")


def _client():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
        return anthropic.Anthropic()
    except Exception:
        return None


# ---------------------------------------------------------------- (a) record parsing
PART_PATTERNS = [
    ("oil", r"\boil\b|oil\s*chg|lube"),
    ("brakes", r"brake|pads?\b|rotor"),
    ("tires", r"\btires?\b|rotation|alignment"),
    ("battery", r"battery"),
    ("transmission", r"transmission|trans\.? fluid"),
    ("coolant", r"coolant|radiator|thermostat|water pump"),
    ("suspension", r"suspension|strut|shock|control arm|tie rod"),
    ("engine", r"engine|spark plug|timing|head gasket|misfire"),
    ("body", r"bumper|fender|door|panel|paint|collision|windshield"),
    ("a/c", r"\ba/?c\b|compressor"),
]
ACTION_PATTERNS = [
    ("replace", r"replac|new |install|swap"),
    ("repair", r"repair|fix|rebuild|resurface"),
    ("service", r"change|chg|service|flush|rotat|top[- ]?off|inspect|check"),
]
MONTHS = {m: i + 1 for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split())}


def _parse_date(s: str):
    s = s.strip()
    m = re.search(r"(\d{4})-(\d{1,2})(?:-(\d{1,2}))?", s)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3) or 1))
    m = re.search(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", s)
    if m:
        y = int(m.group(3)); y += 2000 if y < 100 else 0
        return date(y, int(m.group(1)), int(m.group(2)))
    m = re.search(r"\b([A-Za-z]{3})[a-z]*\.?,?\s+(\d{4})", s)
    if m and m.group(1).lower() in MONTHS:
        return date(int(m.group(2)), MONTHS[m.group(1).lower()], 1)
    return None


def _parse_miles(s: str):
    m = re.search(r"(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(k\b|mi\b|miles?\b)", s, re.I)
    if not m:
        return None
    n = float(m.group(1).replace(",", ""))
    return int(n * 1000) if m.group(2).lower() == "k" else int(n)


def regex_parse(raw: str) -> list[dict]:
    events = []
    for line in [l for l in raw.splitlines() if l.strip()]:
        d, mi = _parse_date(line), _parse_miles(line)
        low = line.lower()
        part = next((p for p, pat in PART_PATTERNS if re.search(pat, low)), "other")
        action = next((a for a, pat in ACTION_PATTERNS if re.search(pat, low)), "service")
        if d:
            events.append({"date": d.isoformat(), "mileage": mi, "part": part, "action": action, "source_text": line.strip()})
    return events


def parse_service_records(raw: str) -> dict:
    """Return {'events': [...], 'mode': 'claude'|'offline-regex'}."""
    client = _client()
    if client and raw.strip():
        try:
            msg = client.messages.create(
                model=PARSE_MODEL, max_tokens=2000,
                system=("You convert messy vehicle service-history text into JSON. Output ONLY a JSON array. "
                        "Each element: {\"date\": \"YYYY-MM-DD\", \"mileage\": int|null, \"part\": one of "
                        "[oil,brakes,tires,battery,transmission,coolant,suspension,engine,body,a/c,other], "
                        "\"action\": one of [service,replace,repair]}. One element per distinct service event. "
                        "If only a month is given use day 01. Do not invent events."),
                messages=[{"role": "user", "content": raw}],
            )
            txt = msg.content[0].text.strip()
            txt = re.sub(r"^```(?:json)?|```$", "", txt, flags=re.M).strip()
            events = json.loads(txt)
            for e in events:
                e.setdefault("source_text", "")
            return {"events": events, "mode": "claude"}
        except Exception as ex:  # fall back, never fail the pipeline on the LLM
            return {"events": regex_parse(raw), "mode": f"offline-regex (Claude call failed: {type(ex).__name__})"}
    return {"events": regex_parse(raw), "mode": "offline-regex"}


# ---------------------------------------------------------------- (b) explanation
def _template_explanation(ctx: dict) -> str:
    r = ctx["decision"]
    if ctx["gate_failed"]:
        gates = ", ".join(g["label"].lower() for g in ctx["gate_failed"])
        base = (f"This vehicle failed a hard safety/title rule ({gates}), so the pricing model was skipped and the car was "
                f"routed straight to the salvage path. Sending it to a retail auction would likely have ended in a failed sale and a repeat listing.")
        if ctx.get("export"):
            ex = ctx["export"]
            base += (f" Export check: eligible for {ex['best']['destination']} (buyer demand {ex['best']['demand']:.1f}), so it goes to the international auction."
                     if ex["eligible"] else " Export check: no destination currently accepts this vehicle, so it goes to the domestic salvage auction.")
        return base
    drivers = "; ".join(f"{d['label']} {'pushed the value up' if d['shap'] > 0 else 'pulled the value down'} (about ${abs(d['shap']):,.0f} vs. an average car)" for d in ctx["drivers"][:3])
    nets = ctx["nets"]
    txt = (f"Recommended route: {r['route_label']}, with an expected net of ${nets[r['route']]:,.0f}. "
           f"Main factors: {drivers}. ")
    if ctx["needs_review"]:
        txt += (f"The model's confidence is {ctx['confidence_pct']:.0f}% and the runner-up ({ctx['runner_up_label']}, ${nets[ctx['runner_up']]:,.0f}) is only {ctx['margin_pct']:.1f}% behind "
                f"({ctx['review_reason']}), so a condition specialist will validate the decision.")
    else:
        txt += f"It beats the runner-up ({ctx['runner_up_label']}) by {ctx['margin_pct']:.0f}% with {ctx['confidence_pct']:.0f}% model confidence, so no human review is needed."
    return txt


def explain_decision(ctx: dict) -> dict:
    client = _client()
    if client:
        try:
            msg = client.messages.create(
                model=EXPLAIN_MODEL, max_tokens=400,
                system=("You write a short, plain-English explanation (3-5 sentences, no jargon, no markdown) of a used-car routing "
                        "decision for a dealer. The decision was ALREADY made by a rules engine and a pricing model; you only explain it. "
                        "Use only the facts in the JSON. Never change or second-guess the decision. Mention if a specialist review is required."),
                messages=[{"role": "user", "content": json.dumps(ctx, default=str)}],
            )
            return {"text": msg.content[0].text.strip(), "mode": "claude"}
        except Exception as ex:
            return {"text": _template_explanation(ctx), "mode": f"offline-template (Claude call failed: {type(ex).__name__})"}
    return {"text": _template_explanation(ctx), "mode": "offline-template"}
