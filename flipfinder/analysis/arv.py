"""ARV estimation from sold comps + renovation cost heuristic."""
import math

import numpy as np

MILES_1_5_LAT = 1.5 / 69.0


def _haversine_miles(lat1, lng1, lat2, lng2):
    r = 3958.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# Attached sales carry a much lower $/sqft than detached ones; leaving them in the
# comp set drags ARV down and hides real spread. Redfin types them as single family,
# so the unit designator in the address is the only signal available here.
ATTACHED_SOLD = "AND address NOT LIKE '% unit %' AND address NOT LIKE '% apt %'"


def _comps(conn, listing):
    sqft = listing["sqft"]
    lo, hi = sqft * 0.75, sqft * 1.25
    rows = []
    if listing["tract"]:
        rows = conn.execute(
            "SELECT ppsf, lat, lng FROM sold "
            f"WHERE tract=? AND ppsf IS NOT NULL AND sqft BETWEEN ? AND ? {ATTACHED_SOLD}",
            (listing["tract"], lo, hi),
        ).fetchall()
    if len(rows) < 3 and listing["lat"] and listing["lng"]:
        dlat = MILES_1_5_LAT
        dlng = MILES_1_5_LAT / max(0.2, math.cos(math.radians(listing["lat"])))
        cands = conn.execute(
            "SELECT ppsf, lat, lng FROM sold "
            "WHERE ppsf IS NOT NULL AND sqft BETWEEN ? AND ? "
            f"AND lat BETWEEN ? AND ? AND lng BETWEEN ? AND ? {ATTACHED_SOLD}",
            (lo, hi, listing["lat"] - dlat, listing["lat"] + dlat,
             listing["lng"] - dlng, listing["lng"] + dlng),
        ).fetchall()
        rows = [
            c for c in cands
            if c["lat"] and c["lng"]
            and _haversine_miles(listing["lat"], listing["lng"], c["lat"], c["lng"]) <= 1.5
        ]
    return [r["ppsf"] for r in rows]


def _reno_tier(listing):
    cond = listing["condition_score"]
    if cond is not None:
        if cond <= 3:
            return "gut"
        if cond <= 6:
            return "medium"
        return "light"
    distress = listing["distress_score"] or 0.0
    return "medium" if distress >= 0.5 else "light"


def estimate(conn, cfg, listing):
    ppsfs = _comps(conn, listing)
    n = len(ppsfs)
    if n == 0:
        return None
    arr = np.array(ppsfs)
    p75 = float(np.percentile(arr, 75))
    arv = p75 * listing["sqft"]
    cv = float(np.std(arr) / np.mean(arr)) if n >= 2 else 0.35
    confidence = min(1.0, n / 5.0) * max(0.0, 1.0 - cv / 0.35)

    tier = _reno_tier(listing)
    reno_cost = cfg["reno_cost_per_sqft"][tier] * listing["sqft"]
    carry = cfg["scoring"]["carry_closing_pct"] * arv
    spread = arv - listing["price"] - reno_cost - carry
    return {
        "arv": arv,
        "reno_tier": tier,
        "reno_cost": reno_cost,
        "spread": spread,
        "margin": spread / arv,
        "confidence": confidence,
        "comp_count": n,
        "comp_p75_ppsf": p75,
    }
