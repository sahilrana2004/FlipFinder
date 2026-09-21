"""ARV estimation from sold comps + renovation cost heuristic."""
import math
import re
from datetime import date

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

# close_ppsf, never ppsf: sold.ppsf is the ASKING price per sqft (Texas is a
# non-disclosure state), so comps built from it value a house in list dollars while
# every target is a close price. Sales with no recovered close price are excluded
# rather than mixed in at the wrong currency.
COMP_FIELDS = "close_ppsf, lat, lng, url, address, sold_date, sqft, year_built, beds"


def _normalize_address(addr):
    return re.sub(r"\s+", " ", (addr or "").strip().lower())


def _exclude_self(rows, exclude_url, exclude_addr):
    """Drop the target's own sale (and any comp at the same address) from a
    comp set — needed so the backtest doesn't score a sale against itself."""
    if not exclude_url:
        return rows
    return [
        r for r in rows
        if r["url"] != exclude_url and _normalize_address(r["address"]) != exclude_addr
    ]


def comp_rows(conn, listing, as_of=None, exclude_url=None):
    """Full comp rows — the backtest needs url/address/sold_date for its leakage
    self-check, and comp weighting needs sqft/year_built/beds."""
    sqft = listing["sqft"]
    lo, hi = sqft * 0.75, sqft * 1.25
    date_sql, date_params = "", []
    if as_of:
        date_sql = " AND sold_date IS NOT NULL AND sold_date < ?"
        date_params = [as_of]
    exclude_addr = _normalize_address(listing["address"]) if exclude_url else None

    rows = []
    if listing["tract"]:
        rows = conn.execute(
            f"SELECT {COMP_FIELDS} FROM sold "
            f"WHERE tract=? AND close_ppsf IS NOT NULL AND sqft BETWEEN ? AND ? "
            f"{ATTACHED_SOLD}{date_sql}",
            (listing["tract"], lo, hi, *date_params),
        ).fetchall()
        rows = _exclude_self(rows, exclude_url, exclude_addr)
    if len(rows) < 3 and listing["lat"] and listing["lng"]:
        dlat = MILES_1_5_LAT
        dlng = MILES_1_5_LAT / max(0.2, math.cos(math.radians(listing["lat"])))
        cands = conn.execute(
            f"SELECT {COMP_FIELDS} FROM sold "
            "WHERE close_ppsf IS NOT NULL AND sqft BETWEEN ? AND ? "
            f"AND lat BETWEEN ? AND ? AND lng BETWEEN ? AND ? {ATTACHED_SOLD}{date_sql}",
            (lo, hi, listing["lat"] - dlat, listing["lat"] + dlat,
             listing["lng"] - dlng, listing["lng"] + dlng, *date_params),
        ).fetchall()
        cands = _exclude_self(cands, exclude_url, exclude_addr)
        rows = [
            c for c in cands
            if c["lat"] and c["lng"]
            and _haversine_miles(listing["lat"], listing["lng"], c["lat"], c["lng"]) <= 1.5
        ]
    return rows


def _comps(conn, listing, as_of=None, exclude_url=None):
    return [r["close_ppsf"] for r in comp_rows(conn, listing, as_of=as_of, exclude_url=exclude_url)]


def _days_between(iso_a, iso_b):
    try:
        a = date.fromisoformat(iso_a)
        b = date.fromisoformat(iso_b)
    except (TypeError, ValueError):
        return None
    return abs((b - a).days)


def comp_weights(listing, rows, as_of=None):
    """Similarity weights: sqft+tract alone can't separate a gut job from retail,
    which is what drives the sub-$250k overprediction. Age, bed count, distance
    and sale recency all narrow the comp set toward genuinely like properties."""
    newest = max((r["sold_date"] for r in rows if r["sold_date"]), default=None)
    ref_date = as_of or newest
    weights = []
    for r in rows:
        w = 1.0
        if listing["sqft"] and r["sqft"]:
            ratio = r["sqft"] / listing["sqft"]
            w *= math.exp(-(((ratio - 1.0) / 0.15) ** 2) / 2)
        if listing["year_built"] and r["year_built"]:
            dy = abs(r["year_built"] - listing["year_built"])
            w *= math.exp(-((dy / 20.0) ** 2) / 2)
        if listing["beds"] and r["beds"]:
            db = abs(r["beds"] - listing["beds"])
            w *= 1.0 if db < 0.5 else (0.7 if db < 1.5 else 0.5)
        if listing["lat"] and listing["lng"] and r["lat"] and r["lng"]:
            d = _haversine_miles(listing["lat"], listing["lng"], r["lat"], r["lng"])
            w *= 1.0 / (1.0 + (d / 0.5) ** 2)
        if ref_date and r["sold_date"]:
            days = _days_between(r["sold_date"], ref_date)
            if days is not None:
                w *= 0.5 ** (days / 365.0)
        weights.append(max(w, 1e-9))
    return weights


def weighted_percentile(values, weights, q):
    order = np.argsort(values)
    v = np.asarray(values, dtype=float)[order]
    w = np.asarray(weights, dtype=float)[order]
    cum = np.cumsum(w)
    if cum[-1] <= 0:
        return float(np.percentile(v, q * 100))
    cutoff = q * cum[-1]
    return float(v[int(np.searchsorted(cum, cutoff))])


def comp_ppsf(listing, rows, percentile, weighted, as_of=None):
    ppsfs = [r["close_ppsf"] for r in rows]
    if not ppsfs:
        return None
    if not weighted:
        return float(np.percentile(np.array(ppsfs), percentile * 100))
    return weighted_percentile(ppsfs, comp_weights(listing, rows, as_of=as_of), percentile)


def tract_ppsf_p90(conn, tract):
    """Guard rail: an ARV implying a $/sqft above almost everything the tract has
    ever sold for is a comp-matching failure, not a find."""
    if not tract:
        return None
    vals = [
        r["close_ppsf"] for r in conn.execute(
            "SELECT close_ppsf FROM sold WHERE tract=? AND close_ppsf IS NOT NULL", (tract,)
        )
    ]
    if len(vals) < 5:
        return None
    return float(np.percentile(np.array(vals), 90))


def max_offer(arv, reno_cost, cfg):
    """The most this house can be paid for and still clear the target margin.

    margin is (arv - price - reno - carry) / arv, so margin >= target_margin is
    exactly price <= arv - reno - carry - target_margin * arv. Deriving the offer
    from the same three terms the margin uses keeps the headline and the margin from
    ever disagreeing: price == max_offer is margin == target_margin, to the cent."""
    s = cfg["scoring"]
    return arv - reno_cost - s["carry_closing_pct"] * arv - s["target_margin"] * arv


def _reno_tier(listing):
    # Photos flatter; the AI's reno_scope also reads the remarks. A human-corrected
    # condition still wins, since the scope was judged alongside the AI's condition.
    if listing["reno_scope"] and listing["condition_source"] != "human":
        return listing["reno_scope"]
    cond = listing["condition_score"]
    if cond is not None:
        if cond <= 3:
            return "gut"
        if cond <= 6:
            return "medium"
        return "light"
    distress = listing["distress_score"] or 0.0
    return "medium" if distress >= 0.5 else "light"


def estimate(conn, cfg, listing, as_of=None, exclude_url=None):
    """ARV is the AVM's price for this house in renovated condition (see avm.py);
    the comps still set confidence and the tract-p90 guard."""
    from . import avm

    rows = comp_rows(conn, listing, as_of=as_of, exclude_url=exclude_url)
    n = len(rows)
    if n == 0:
        return None
    s = cfg["scoring"]
    ppsf = comp_ppsf(listing, rows, s["arv_percentile"], s["arv_weighted"], as_of=as_of)
    arv, feats = avm.predict_arv(conn, listing, as_of=as_of, exclude_url=exclude_url)

    ppsfs = [r["close_ppsf"] for r in rows]
    arr = np.array(ppsfs)
    cv = float(np.std(arr) / np.mean(arr)) if n >= 2 else 0.35
    confidence = min(1.0, n / 5.0) * max(0.0, 1.0 - cv / 0.35)

    p90 = tract_ppsf_p90(conn, listing["tract"])
    arv_ppsf = arv / listing["sqft"]
    above_tract_p90 = bool(p90 and arv_ppsf > p90)

    tier = _reno_tier(listing)
    reno_cost = cfg["reno_cost_per_sqft"][tier] * listing["sqft"]
    carry = s["carry_closing_pct"] * arv
    spread = arv - listing["price"] - reno_cost - carry
    offer = max_offer(arv, reno_cost, cfg)
    return {
        "arv": arv,
        "max_offer": offer,
        # What the ask has to come down by to become that offer. Negative would mean
        # the house is already cheap enough; no listing here is.
        "offer_discount": (listing["price"] - offer) / listing["price"],
        "as_is_value": feats["_as_is"],
        "arv_floored": feats["_arv_floored"],
        "reno_tier": tier,
        "reno_cost": reno_cost,
        "spread": spread,
        "margin": spread / arv,
        "confidence": confidence,
        "comp_count": n,
        "comp_ppsf": ppsf,
        "arv_ppsf": arv_ppsf,
        "tract_p90_ppsf": p90,
        "above_tract_p90": above_tract_p90,
    }
