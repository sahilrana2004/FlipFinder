"""Investment score: multiplicative combination of margin, confidence,
liquidity, distress, and a size/age multiplier. Weights are exponents so
they can be fitted against Sahil's labels (see fitter.py).

The score is absolute. It used to be min-maxed across whatever happened to be
scored that run, which forced the best listing to exactly 1.00 however bad it was
and the worst to 0.00 however good, and made deal_threshold mean nothing fixed.
Margin now goes through a logistic centred on the target margin, and the product of
the four factors is raised to 1/(sum of the exponents) — a weighted geometric mean.
That is a fixed monotone rescale, not a renormalisation: it leaves the ranking the
raw product gives exactly as it was, and puts the score on the same 0-1 scale as
its factors, so a score of 0.5 means "as good as a listing sitting at the target
margin with 0.5 confidence, 0.5 liquidity and 0.5 distress" — whatever the
exponents are, since the geometric mean of four 0.5s is 0.5 for any weights.
"""
import json
import math
import re

import numpy as np

from ..db import current_weights
from . import arv as arv_mod

FEATURE_KEYS = ("margin", "confidence", "liquidity", "distress")


def margin_factor(margin, cfg):
    """Logistic on margin, centred on the target margin. The old score took
    max(margin, 0.001), which flattened every negative-margin listing onto the same
    floor and left them to be ranked by the other three factors — a house that loses
    5% and one that loses 50% scored identically on the term that matters most."""
    s = cfg["scoring"]
    z = s["margin_steepness"] * (margin - s["target_margin"])
    if z < -700:                       # exp overflows below this; the answer is 0
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def combine(feats, weights, cfg):
    m = max(margin_factor(feats["margin"], cfg), 1e-6)
    c = max(feats["confidence"], 0.01)
    l = max(feats["liquidity"], 0.01)
    d = max(feats["distress"], 0.01)
    raw = (
        (m ** weights["margin"])
        * (c ** weights["confidence"])
        * (l ** weights["liquidity"])
        * (d ** weights["distress"])
    )
    total = sum(weights[k] for k in FEATURE_KEYS)
    return raw ** (1.0 / total) * feats["size_mult"] if total > 0 else 0.0


def _pct_rank(values):
    """Mid-rank in [0, 1]: the share of tracts this one beats, counting ties half.
    A rank can't clip, which is the whole point — both liquidity terms used to
    saturate at 1.0 for a quarter of the tracts."""
    arr = np.array(list(values.values()), dtype=float)
    n = len(arr)
    if n < 2:
        return {k: 0.5 for k in values}
    order = np.argsort(arr)
    ranks = {}
    for k, v in values.items():
        lower = int(np.searchsorted(arr[order], v, side="left"))
        equal = int(np.searchsorted(arr[order], v, side="right")) - lower
        ranks[k] = (lower + 0.5 * equal) / n
    return ranks


def liquidity_index(conn):
    """Liquidity per tract, as percentile ranks across every tract with stats.

    Both terms used to be ratios with a ceiling: sold_count over a 3-year window
    against a current active count, divided by 2 and clipped at 1, and 45/median_dom
    clipped at 1 whenever the tract's median DOM was under 45 days — which, at 1-6
    days, it almost always is. A quarter of tracts scored exactly 1.000 and every
    scored listing did. Ranks keep the ordering and spend the whole 0-1 range."""
    rows = conn.execute(
        "SELECT tract, sold_count_12m, active_count, median_dom FROM area_stats"
    ).fetchall()
    if not rows:
        return {}
    pace = {r["tract"]: (r["sold_count_12m"] or 0) / max(r["active_count"] or 0, 1)
            for r in rows}
    speed = {r["tract"]: -r["median_dom"] for r in rows if r["median_dom"]}
    pace_rank = _pct_rank(pace)
    speed_rank = _pct_rank(speed)
    # A tract with no active listing carrying a DOM gets the middle of the range
    # rather than a guess in either direction.
    return {t: 0.5 * pace_rank[t] + 0.5 * speed_rank.get(t, 0.5) for t in pace}


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
    liquidity = liquidity_index(conn)
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
            "liquidity": liquidity.get(listing["tract"], 0.3),
            "distress": listing["distress_score"] or 0.0,
            "size_mult": _size_mult(listing, cfg),
        }
        results.append((listing["id"], feats, est, combine(feats, weights, cfg)))

    for listing_id, feats, est, score in results:
        components = {**feats, "arv": est["arv"], "reno_cost": est["reno_cost"],
                      "max_offer": est["max_offer"],
                      "offer_discount": est["offer_discount"],
                      "reno_tier": est["reno_tier"], "spread": est["spread"],
                      "comp_count": est["comp_count"],
                      "comp_ppsf": est["comp_ppsf"],
                      "arv_ppsf": est["arv_ppsf"],
                      "as_is_value": est["as_is_value"],
                      "arv_floored": est["arv_floored"],
                      "margin_factor": margin_factor(feats["margin"], cfg),
                      "tract_p90_ppsf": est["tract_p90_ppsf"],
                      "above_tract_p90": est["above_tract_p90"]}
        conn.execute(
            """INSERT OR REPLACE INTO scores
               (listing_id, scorer_version, score, raw_score, margin, arv, reno_cost,
                spread, max_offer, offer_discount, confidence, liquidity, distress,
                size_mult, components)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (listing_id, version, min(score, 1.0), score, feats["margin"], est["arv"],
             est["reno_cost"], est["spread"], est["max_offer"], est["offer_discount"],
             feats["confidence"], feats["liquidity"], feats["distress"],
             feats["size_mult"], json.dumps(components)),
        )
    conn.commit()
    return len(results), version
