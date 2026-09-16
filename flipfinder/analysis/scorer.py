"""Investment score: multiplicative combination of margin, confidence,
liquidity, distress, and a size/age multiplier. Weights are exponents so
they can be fitted against Sahil's labels (see fitter.py).
"""
import json
import re

from ..db import current_weights
from . import arv as arv_mod

FEATURE_KEYS = ("margin", "confidence", "liquidity", "distress")


def combine(feats, weights):
    m = max(feats["margin"], 0.001)
    c = max(feats["confidence"], 0.01)
    l = max(feats["liquidity"], 0.01)
    d = max(feats["distress"], 0.01)
    return (
        (m ** weights["margin"])
        * (c ** weights["confidence"])
        * (l ** weights["liquidity"])
        * (d ** weights["distress"])
        * feats["size_mult"]
    )


def _liquidity(conn, tract):
    if not tract:
        return 0.3
    s = conn.execute(
        "SELECT sold_count, active_count, median_dom FROM area_stats WHERE tract=?",
        (tract,),
    ).fetchone()
    if not s:
        return 0.3
    solds = s["sold_count"] or 0
    actives = s["active_count"] or 0
    ratio_f = min(1.0, (solds / max(actives, 1)) / 2.0)
    dom_f = min(1.0, 45.0 / s["median_dom"]) if s["median_dom"] else 0.5
    return 0.5 * ratio_f + 0.5 * dom_f


def _size_mult(listing, cfg):
    ss = cfg["scoring"]["size_sweet_spot"]
    mult = 1.0
    if (
        listing["sqft"]
        and ss["min_sqft"] <= listing["sqft"] <= ss["max_sqft"]
        and listing["beds"] in ss["beds"]
        and (listing["baths"] or 0) >= ss["min_baths"]
    ):
        mult = ss["bonus"]
    elif listing["sqft"] and (listing["sqft"] < 800 or listing["sqft"] > 2600):
        mult = 0.9
    if listing["year_built"] and listing["year_built"] < 1900:
        mult *= 0.9
    return mult


# Redfin labels attached housing "Single Family Residential" in the CSV export, so
# property type can't filter it. Left in, a condo's low $/sqft reads as a deep
# discount against detached comps and fabricates an ARV roughly double reality.
UNIT_IN_ADDRESS = re.compile(r"\b(unit|apt|ste)\b", re.I)
ATTACHED_IN_REMARKS = re.compile(r"\b(condo(minium)?|townhome|townhouse)\b", re.I)


def _is_attached(listing):
    return bool(
        UNIT_IN_ADDRESS.search(listing["address"] or "")
        or ATTACHED_IN_REMARKS.search(listing["remarks"] or "")
    )


def _in_buy_box(listing, cfg):
    b = cfg["buy_box"]
    return (
        listing["price"] and b["min_price"] <= listing["price"] <= b["max_price"]
        and listing["sqft"] and b["min_sqft"] <= listing["sqft"] <= b["max_sqft"]
        and (listing["beds"] or 0) >= b["min_beds"]
        and not _is_attached(listing)
    )


def score_all(conn, cfg, weights=None, version=None):
    if weights is None:
        version, weights = current_weights(conn, cfg)
    conn.execute("DELETE FROM scores")
    results = []
    for listing in conn.execute("SELECT * FROM listings WHERE active=1"):
        if not _in_buy_box(listing, cfg):
            continue
        est = arv_mod.estimate(conn, cfg, listing)
        if est is None:
            continue
        feats = {
            "margin": est["margin"],
            "confidence": est["confidence"],
            "liquidity": _liquidity(conn, listing["tract"]),
            "distress": listing["distress_score"] or 0.0,
            "size_mult": _size_mult(listing, cfg),
        }
        raw = combine(feats, weights)
        results.append((listing["id"], feats, est, raw))
    if not results:
        conn.commit()
        return 0, version

    raws = [r[3] for r in results]
    lo, hi = min(raws), max(raws)
    span = (hi - lo) or 1.0
    for listing_id, feats, est, raw in results:
        norm = (raw - lo) / span
        components = {**feats, "arv": est["arv"], "reno_cost": est["reno_cost"],
                      "reno_tier": est["reno_tier"], "spread": est["spread"],
                      "comp_count": est["comp_count"],
                      "comp_ppsf": est["comp_ppsf"],
                      "tract_p90_ppsf": est["tract_p90_ppsf"],
                      "above_tract_p90": est["above_tract_p90"]}
        conn.execute(
            """INSERT OR REPLACE INTO scores
               (listing_id, scorer_version, score, raw_score, margin, arv, reno_cost,
                spread, confidence, liquidity, distress, size_mult, components)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (listing_id, version, norm, raw, feats["margin"], est["arv"],
             est["reno_cost"], est["spread"], feats["confidence"], feats["liquidity"],
             feats["distress"], feats["size_mult"], json.dumps(components)),
        )
    conn.commit()
    return len(results), version
