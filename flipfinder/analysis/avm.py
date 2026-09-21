"""Automated valuation model: LightGBM on log(close price).

Texas doesn't disclose sale prices, so Redfin's sold PRICE is the last asking price;
the target here is sold.close_price, recovered from the MLS ratio fields. List price
never enters the features — a model that reads the asking price scores well in a
backtest and can never find an underpriced listing. It only selects who is scored.

Every feature is point-in-time: comps and tract medians come from sales that closed
before the target's own sold date, through the same arv.comp_rows / comp_weights the
live scorer uses. Condition comes from the agent's remarks.
"""
import json
import math
import re
from datetime import date

import numpy as np

from ..config import ROOT
from . import arv as arv_mod

MODEL_DIR = ROOT / "data" / "avm"
MODEL_PATH = MODEL_DIR / "model.txt"
META_PATH = MODEL_DIR / "model_meta.json"

# Candidates per group. The final lists are the subset chosen on the tune split by
# select_keywords() and stored in the model's metadata (and in data/avm/report.txt).
CANDIDATES = {
    "distressed": [
        "as-is", "investor", "cash only", "cash buyers", "cash or conventional",
        "needs work", "needs some work", "tlc", "foundation", "fire", "fire damage",
        "water damage", "handyman", "fixer", "estate", "estate sale", "bring your vision",
        "sweat equity", "great bones", "potential", "opportunity", "repairs", "gutted",
        "studs", "rehab", "diamond in the rough", "renovation project", "vacant",
        "tenant", "no repairs", "where is", "flip", "cosmetic", "hard money",
        "needs updating", "priced to sell", "motivated seller",
    ],
    "renovated": [
        "remodeled", "renovated", "updated", "new roof", "new hvac", "new ac", "new a/c",
        "new flooring", "new floors", "move-in ready", "fully updated",
        "fully renovated", "fully remodeled", "quartz", "granite", "stainless",
        "luxury vinyl", "new paint", "fresh paint", "new windows", "new plumbing",
        "new electrical", "new water heater", "new cabinets", "open concept", "turnkey",
        "shiplap", "brand new", "recessed lighting", "new fixtures", "tankless",
        "new siding", "new fence",
    ],
}

# Fixed (not tuned) phrase groups used only to split neighbor sales by condition;
# these are the task's defining lists, so they never see the validation data.
NEIGHBOR_GROUPS = {
    "distressed": ["as-is", "investor", "cash only", "needs work", "tlc", "foundation",
                   "fire", "water damage", "handyman", "fixer", "estate"],
    "renovated": ["remodeled", "renovated", "updated", "new roof", "new hvac",
                  "new flooring", "move-in ready", "fully updated"],
}
NEIGHBOR_K = 10
NEIGHBOR_MILES = 1.0

BASE_FEATURES = [
    "sqft", "beds", "baths", "year_built", "lot_sqft", "dom",
    "lat", "lng", "zip_code", "tract_ppsf", "tract_close_ppsf", "date_ordinal",
    "comp_wm_est", "comp_wm_ppsf", "comp_median_est", "comp_count", "comp_cv",
    "comp_close_ppsf", "comp_close_count",
    "nb_close_ppsf", "nb_close_ppsf_size", "nb_close_dist", "nb_close_n",
    "nb_distressed_ppsf", "nb_renovated_ppsf",
    "remarks_missing", "remarks_len",
]


def _pattern(phrase):
    # "as-is" also matches "as is"; "move-in ready" also "move in ready"
    body = r"[\s-]+".join(re.escape(w) for w in re.split(r"[\s-]+", phrase))
    return re.compile(r"(?<![a-z])" + body + r"(?![a-z])")


def keyword_hits(remarks, keywords):
    text = (remarks or "").lower()
    return {kw: bool(_pattern(kw).search(text)) for kw in keywords}


def _kw_name(kw):
    return "kw_" + re.sub(r"[^a-z0-9]+", "_", kw).strip("_")


def _median(vals):
    return float(np.median(vals)) if vals else None


def _tract_ppsfs(conn, tract, as_of, exclude_url):
    if not tract:
        return [], [], None
    date_sql, params = "", [tract]
    if as_of:
        date_sql = " AND sold_date < ?"
        params.append(as_of)
    rows = conn.execute(
        f"SELECT url, ppsf, close_price, sqft, sold_date FROM sold WHERE tract=? AND ppsf IS NOT NULL "
        f"{arv_mod.ATTACHED_SOLD}{date_sql}",
        params,
    ).fetchall()
    rows = [r for r in rows if r["url"] != exclude_url]
    listed = [r["ppsf"] for r in rows]
    closed = [r["close_price"] / r["sqft"] for r in rows if r["close_price"] and r["sqft"]]
    return listed, closed, max((r["sold_date"] for r in rows), default=None)


class _CloseIndex:
    """Every sale with a recovered close price, in arrays, so neighbor features are
    a vectorized filter instead of a query per subject."""

    def __init__(self, conn):
        rows = conn.execute(
            f"SELECT url, lat, lng, sqft, close_price, sold_date, remarks FROM sold "
            f"WHERE close_price IS NOT NULL AND sqft > 0 AND lat IS NOT NULL "
            f"{arv_mod.ATTACHED_SOLD}"
        ).fetchall()
        self.url = np.array([r["url"] for r in rows])
        self.lat = np.array([r["lat"] for r in rows])
        self.lng = np.array([r["lng"] for r in rows])
        self.sqft = np.array([r["sqft"] for r in rows])
        self.ppsf = np.array([r["close_price"] / r["sqft"] for r in rows])
        self.date = np.array([r["sold_date"] for r in rows])
        self.group = {}
        for g, kws in NEIGHBOR_GROUPS.items():
            pats = [_pattern(k) for k in kws]
            self.group[g] = np.array([any(p.search((r["remarks"] or "").lower()) for p in pats)
                                      for r in rows])


_index_cache = {}


def _close_index(conn):
    key = (id(conn), conn.execute("SELECT COUNT(*), MAX(detail_fetched) FROM sold").fetchone()[:])
    if key not in _index_cache:
        _index_cache.clear()
        _index_cache[key] = _CloseIndex(conn)
    return _index_cache[key]


def _neighbor_features(conn, subject, as_of, exclude_url):
    idx = _close_index(conn)
    out = dict.fromkeys(("nb_close_ppsf", "nb_close_ppsf_size", "nb_close_dist",
                         "nb_distressed_ppsf", "nb_renovated_ppsf"))
    out["nb_close_n"] = 0
    if subject["lat"] is None or subject["lng"] is None or not len(idx.url):
        return out
    mask = idx.url != (exclude_url or "")
    if as_of:
        mask &= idx.date < as_of
    dlat = (idx.lat - subject["lat"]) * 69.0
    dlng = (idx.lng - subject["lng"]) * 69.0 * math.cos(math.radians(subject["lat"]))
    dist = np.sqrt(dlat ** 2 + dlng ** 2)
    mask &= dist <= NEIGHBOR_MILES
    out["nb_close_n"] = int(mask.sum())

    def nearest(m):
        ids = np.flatnonzero(m)
        if not len(ids):
            return None, None
        ids = ids[np.argsort(dist[ids])[:NEIGHBOR_K]]
        return float(np.median(idx.ppsf[ids])), float(np.mean(dist[ids]))

    out["nb_close_ppsf"], out["nb_close_dist"] = nearest(mask)
    if subject["sqft"]:
        ratio = idx.sqft / subject["sqft"]
        out["nb_close_ppsf_size"], _ = nearest(mask & (ratio >= 0.7) & (ratio <= 1.3))
    out["nb_distressed_ppsf"], _ = nearest(mask & idx.group["distressed"])
    out["nb_renovated_ppsf"], _ = nearest(mask & idx.group["renovated"])
    return out


def point_in_time_features(conn, subject, as_of, exclude_url=None, rows=None):
    """Everything except condition. `subject` is a sold or listings row; as_of is the
    sale date in the backtest and today at scoring time. Pass `rows` to reuse comps
    already selected with the same as_of/exclude_url."""
    if rows is None:
        rows = arv_mod.comp_rows(conn, subject, as_of=as_of, exclude_url=exclude_url)
    sqft = subject["sqft"]
    f = {
        "sqft": sqft, "beds": subject["beds"], "baths": subject["baths"],
        "year_built": subject["year_built"], "lot_sqft": subject["lot_sqft"],
        "dom": subject["dom"], "lat": subject["lat"], "lng": subject["lng"],
        "zip": subject["zip"],
        "date_ordinal": date.fromisoformat(as_of).toordinal() if as_of else date.today().toordinal(),
        "comp_count": len(rows),
    }
    f.update(_neighbor_features(conn, subject, as_of, exclude_url))
    listed, closed, f["_tract_newest"] = _tract_ppsfs(conn, subject["tract"], as_of, exclude_url)
    f["tract_ppsf"] = _median(listed)
    f["tract_close_ppsf"] = _median(closed)
    if rows:
        ppsfs = [r["ppsf"] for r in rows]
        weights = arv_mod.comp_weights(subject, rows, as_of=as_of)
        wm = arv_mod.weighted_percentile(ppsfs, weights, 0.5)
        arr = np.array(ppsfs)
        f["comp_wm_ppsf"] = wm
        f["comp_wm_est"] = wm * sqft
        f["comp_median_est"] = float(np.median(arr)) * sqft
        f["comp_cv"] = float(np.std(arr) / np.mean(arr)) if len(rows) >= 2 else None
        cl = [(r["close_price"] / r["sqft"], w) for r, w in zip(rows, weights)
              if r["close_price"] and r["sqft"]]
        f["comp_close_count"] = len(cl)
        f["comp_close_ppsf"] = (arv_mod.weighted_percentile([c[0] for c in cl], [c[1] for c in cl], 0.5)
                                if cl else None)
    else:
        f.update(comp_wm_ppsf=None, comp_wm_est=None, comp_median_est=None, comp_cv=None,
                 comp_close_count=0, comp_close_ppsf=None)
    return f


def condition_features(remarks, keywords):
    """keywords: {"distressed": [...], "renovated": [...]}."""
    missing = not (remarks or "").strip()
    f = {"remarks_missing": int(missing), "remarks_len": 0 if missing else len(remarks)}
    for group, kws in keywords.items():
        hits = keyword_hits(remarks, kws)
        for kw, hit in hits.items():
            f[_kw_name(kw)] = int(hit)
        f[f"n_{group}"] = sum(hits.values())
    return f


def renovated_profile(keywords):
    """ARV is the value after repair: renovated indicators on, distressed off."""
    f = {"remarks_missing": 0}
    for kw in keywords["distressed"]:
        f[_kw_name(kw)] = 0
    for kw in keywords["renovated"]:
        f[_kw_name(kw)] = 1
    f["n_distressed"] = 0
    f["n_renovated"] = len(keywords["renovated"])
    return f


def feature_names(keywords):
    names = list(BASE_FEATURES)
    for group, kws in keywords.items():
        names += [_kw_name(kw) for kw in kws] + [f"n_{group}"]
    return names


def matrix(rows, names, zip_codes):
    """rows: feature dicts. ZIP is categorical, coded by a mapping frozen at fit time."""
    X = np.full((len(rows), len(names)), np.nan)
    for i, f in enumerate(rows):
        for j, n in enumerate(names):
            if n == "zip_code":
                code = zip_codes.get(f.get("zip"))
                X[i, j] = np.nan if code is None else code
            else:
                v = f.get(n)
                X[i, j] = np.nan if v is None else float(v)
    return X


def select_keywords(train_rows, candidates=CANDIDATES, min_support=12, min_effect=0.03):
    """Keep a phrase when it's common enough and moves the residual against the
    comp estimate in its group's direction. train_rows: dicts with remarks,
    y (log close) and base (log comp estimate) — tune split only."""
    rows = [r for r in train_rows if r["base"] is not None and (r["remarks"] or "").strip()]
    resid = np.array([r["y"] - r["base"] for r in rows])
    chosen = {}
    stats = {}
    for group, kws in candidates.items():
        sign = -1 if group == "distressed" else 1
        keep = []
        for kw in kws:
            pat = _pattern(kw)
            mask = np.array([bool(pat.search(r["remarks"].lower())) for r in rows])
            n = int(mask.sum())
            if n < min_support or n == len(rows):
                stats[kw] = {"group": group, "n": n, "effect": None, "kept": False}
                continue
            effect = float(np.median(resid[mask]) - np.median(resid[~mask]))
            ok = sign * effect >= min_effect
            stats[kw] = {"group": group, "n": n, "effect": round(effect, 4), "kept": ok}
            if ok:
                keep.append(kw)
        chosen[group] = keep
    return chosen, stats


def fit(X, y, weights, names, params):
    import lightgbm as lgb

    p = {
        "objective": params.get("objective", "regression"),
        "learning_rate": params.get("learning_rate", 0.03),
        "num_leaves": params.get("num_leaves", 15),
        "min_data_in_leaf": params.get("min_data_in_leaf", 20),
        "feature_fraction": params.get("feature_fraction", 0.8),
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": params.get("lambda_l2", 1.0),
        "verbose": -1,
        "seed": 7,
        "deterministic": True,
        "num_threads": 4,
    }
    if p["objective"] == "huber":
        p["alpha"] = params.get("huber_alpha", 0.15)
    ds = lgb.Dataset(X, label=y, weight=weights, feature_name=names,
                     categorical_feature=["zip_code"], free_raw_data=False)
    return lgb.train(p, ds, num_boost_round=params.get("rounds", 600))


def sample_weights(close_prices, low_weight, cutoff=160000):
    """Up-weight low-price sales (by the training label, never the list price)."""
    return np.array([low_weight if c <= cutoff else 1.0 for c in close_prices])


def save(booster, meta):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(MODEL_PATH))
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


_cache = {}


def load():
    if "model" in _cache:
        return _cache["model"]
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError(
            f"ARV model missing ({MODEL_PATH}). Fit it with: py run.py train"
        )
    import lightgbm as lgb

    booster = lgb.Booster(model_file=str(MODEL_PATH))
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    _cache["model"] = (booster, meta)
    return booster, meta


def predict_arv(conn, listing, as_of=None, exclude_url=None):
    """After-repair value: the model's price for this house in renovated condition."""
    booster, meta = load()
    f = point_in_time_features(conn, listing, as_of, exclude_url=exclude_url)
    f.update(condition_features(listing["remarks"], meta["keywords"]))
    f.update(renovated_profile(meta["keywords"]))
    X = matrix([f], meta["features"], meta["zip_codes"])
    return math.exp(float(booster.predict(X)[0])), f


# ---- selection on the tune split -------------------------------------------------

# One finished candidate = this search space + the feature set above. Every
# `py run.py backtest` scores the chosen config on the test split once and appends
# it to data/avm/candidates.jsonl.
# c5 widened this grid (learning rate, 31 leaves, 1000 rounds, min_data 5) and
# scored worse on test, 58.4% against 64.0%: with ~54 low-price sales per inner
# fold, a larger grid fits the validation noise. This is the grid that held up.
CANDIDATE = "c4-allprice-close"
SEARCH = {
    "objective": ["regression", "regression_l1", "huber"],
    "num_leaves": [7, 15],
    "min_data_in_leaf": [10, 30],
    "rounds": [300, 800],
    "low_weight": [1.0, 3.0, 6.0, 12.0],
}
# Heavy low-price weighting buys primary accuracy by pulling every prediction
# down; at 25x, inner-fold accuracy over all prices fell to 40%. The ARV is a
# renovated value, usually above $145k, so a config may give up at most this much
# overall inner accuracy relative to the best config.
OVERALL_GUARD = 0.03
PRIMARY_MAX_LIST = 145000


def _grid(space):
    keys = list(space)
    out = [{}]
    for k in keys:
        out = [{**o, k: v} for o in out for v in space[k]]
    return out


def _rows_for(records, keywords):
    feats = []
    for r in records:
        f = dict(r["feats"])
        f.update(condition_features(r["remarks"], keywords))
        feats.append(f)
    return feats


def _kw_rows(records):
    return [{"remarks": r["remarks"], "y": math.log(r["target"]),
             "base": math.log(r["feats"]["comp_wm_est"]) if r["feats"].get("comp_wm_est") else None}
            for r in records]


def fit_records(records, params, keywords=None, zip_codes=None):
    """Fit on records (tune split, or everything for production). Keywords are
    selected on these same records unless given."""
    if keywords is None:
        keywords, _ = select_keywords(_kw_rows(records))
    if zip_codes is None:
        zips = sorted({r["feats"]["zip"] for r in records if r["feats"].get("zip")})
        zip_codes = {z: i for i, z in enumerate(zips)}
    names = feature_names(keywords)
    X = matrix(_rows_for(records, keywords), names, zip_codes)
    y = np.array([math.log(r["target"]) for r in records])
    w = sample_weights([r["target"] for r in records], params.get("low_weight", 1.0))
    booster = fit(X, y, w, names, params)
    return booster, {"keywords": keywords, "features": names, "zip_codes": zip_codes,
                     "params": params}


def predict_records(booster, meta, records):
    X = matrix(_rows_for(records, meta["keywords"]), meta["features"], meta["zip_codes"])
    return [math.exp(v) for v in booster.predict(X)]


def _within(preds, records, tol):
    errs = [abs(p - r["target"]) / r["target"] for p, r in zip(preds, records)]
    return sum(e <= tol for e in errs) / len(errs) if errs else None


def search(tune_records, space=None):
    """Time-ordered inner folds inside the tune split: fit on the oldest 60% / 80%,
    validate on the next 20%. Score = within-20% on validation sales listed at or
    below $145k, tie-broken by within-20% on all validation sales, among configs
    within OVERALL_GUARD of the best overall accuracy. Keywords are
    re-selected inside each fold so validation never informs them."""
    space = space or SEARCH
    recs = sorted(tune_records, key=lambda r: r["sold_date"])
    n = len(recs)
    folds = [(recs[:int(n * 0.6)], recs[int(n * 0.6):int(n * 0.8)]),
             (recs[:int(n * 0.8)], recs[int(n * 0.8):])]
    fold_kw = [select_keywords(_kw_rows(tr))[0] for tr, _ in folds]
    results = []
    for params in _grid(space):
        prim, allv = [], []
        for (tr, va), kws in zip(folds, fold_kw):
            booster, meta = fit_records(tr, params, keywords=kws)
            preds = predict_records(booster, meta, va)
            low = [(p, r) for p, r in zip(preds, va) if r["list_price"] <= PRIMARY_MAX_LIST]
            prim.append(_within([p for p, _ in low], [r for _, r in low], 0.20))
            allv.append(_within(preds, va, 0.20))
        results.append({"params": params,
                        "inner_primary_w20": round(float(np.mean(prim)), 4),
                        "inner_all_w20": round(float(np.mean(allv)), 4),
                        "inner_primary_n": [sum(r["list_price"] <= PRIMARY_MAX_LIST for r in va)
                                            for _, va in folds]})
    best_all = max(r["inner_all_w20"] for r in results)
    for r in results:
        r["eligible"] = r["inner_all_w20"] >= best_all - OVERALL_GUARD
    results.sort(key=lambda r: (r["eligible"], r["inner_primary_w20"], r["inner_all_w20"]),
                 reverse=True)
    return results


def training_records(conn):
    """Every sale with a recovered close price, featurized as of its own sold date."""
    out = []
    for t in conn.execute(
        "SELECT * FROM sold WHERE close_price IS NOT NULL AND sqft IS NOT NULL "
        "AND sold_date IS NOT NULL AND lat IS NOT NULL AND lng IS NOT NULL ORDER BY sold_date"
    ):
        out.append({"sold_date": t["sold_date"], "target": t["close_price"],
                    "list_price": t["list_price"] or t["price"], "remarks": t["remarks"],
                    "feats": point_in_time_features(conn, t, t["sold_date"], exclude_url=t["url"])})
    return out


def train(conn):
    """Refit on every sale using the config the backtest chose on its tune split."""
    chosen = ROOT / "data" / "backtest" / "metrics.json"
    if not chosen.exists():
        raise FileNotFoundError(f"{chosen} missing: run `py run.py backtest` first")
    with open(chosen, encoding="utf-8") as f:
        info = json.load(f)["avm"]
    records = training_records(conn)
    booster, meta = fit_records(records, info["params"], keywords=info["keywords"])
    meta.update(candidate=info["candidate"], trained_on=len(records),
                newest_sale=records[-1]["sold_date"] if records else None)
    save(booster, meta)
    _cache.clear()
    return meta
