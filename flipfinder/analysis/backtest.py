"""Leakage-safe ARV backtest: for every real sale on disk, predict its price
from comps known as of its own sold_date (production comp selection, so the
backtest can't drift from what the live scorer does), sweep the comp percentile
on the tune split, fit the AVM on the tune split, and score everything on the
untouched test split.

The target is the close price. Texas doesn't disclose sale prices, so Redfin's
sold PRICE is the last asking price; sold.close_price comes from the MLS ratio
fields on the detail page. Sales without a recovered close price can't be scored
and are counted, not silently dropped.

The primary population is test sales LISTED at or below $145k: the app chooses
listings by asking price, so choosing by sale price would select on the answer.

Comps are priced in close dollars (sold.close_ppsf), so the comp baselines predict
in the same currency as the target instead of in asking dollars.
"""
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from ..config import ROOT
from . import arv as arv_mod
from . import avm

TUNE_FRACTION = 0.7
MIN_SAMPLE = 30
BUY_BOX_MAX = 145000
SWEEP = [round(0.20 + 0.05 * i, 2) for i in range(13)]

PRICE_BANDS = (
    ("<150k", None, 150000),
    ("150-250k", 150000, 250000),
    ("250-400k", 250000, 400000),
    (">400k", 400000, None),
)
COMP_BANDS = (
    ("1-2", 1, 2),
    ("3-5", 3, 5),
    ("6+", 6, None),
)


POPULATIONS = (
    ("listed<=145k", lambda r: r["list_price"] <= BUY_BOX_MAX),
    ("sold<=145k", lambda r: r["target"] <= BUY_BOX_MAX),
    ("sold145-250k", lambda r: BUY_BOX_MAX < r["target"] <= 250000),
    ("overall", lambda r: True),
)


def _band(value, bands):
    for name, lo, hi in bands:
        if (lo is None or value >= lo) and (hi is None or value <= hi):
            return name
    return None


def _predict(rec, weighted, q):
    if "pred" in rec and weighted is None:
        return rec["pred"].get(q)
    if not rec["ppsfs"]:
        return None
    if weighted:
        ppsf = arv_mod.weighted_percentile(rec["ppsfs"], rec["weights"], q)
    else:
        ppsf = float(np.percentile(np.array(rec["ppsfs"]), q * 100))
    return ppsf * rec["sqft"]


def _targets(conn):
    return conn.execute(
        "SELECT * FROM sold WHERE price IS NOT NULL AND sqft IS NOT NULL "
        "AND sold_date IS NOT NULL AND lat IS NOT NULL AND lng IS NOT NULL "
        "ORDER BY sold_date, id"
    ).fetchall()


def _build_records(conn, targets):
    """Run production comp selection for every target and check for leakage
    while we still have the raw comp rows (url/address/sold_date) on hand."""
    n_tune = int(len(targets) * TUNE_FRACTION)
    records = []
    violations = 0
    checked = 0

    for i, t in enumerate(targets):
        rows = arv_mod.comp_rows(conn, t, as_of=t["sold_date"], exclude_url=t["url"])
        target_addr = arv_mod._normalize_address(t["address"])

        checked += 1
        for r in rows:
            if r["sold_date"] and r["sold_date"] >= t["sold_date"]:
                violations += 1
            elif r["url"] == t["url"] or arv_mod._normalize_address(r["address"]) == target_addr:
                violations += 1
        feats = avm.point_in_time_features(conn, t, t["sold_date"], exclude_url=t["url"], rows=rows)
        if feats["_tract_newest"] and feats["_tract_newest"] >= t["sold_date"]:
            violations += 1

        records.append({
            "url": t["url"], "address": t["address"], "zip": t["zip"],
            "sold_date": t["sold_date"], "price": t["price"], "sqft": t["sqft"],
            # the CSV's sold PRICE is the last asking price (TX non-disclosure), so it
            # stands in for list_price where the detail page gave none
            "list_price": t["list_price"] or t["price"],
            "target": t["close_price"], "remarks": t["remarks"],
            "comp_count": len(rows), "split": "tune" if i < n_tune else "test",
            "ppsfs": [r["close_ppsf"] for r in rows],
            "weights": arv_mod.comp_weights(t, rows, as_of=t["sold_date"]) if rows else [],
            "feats": feats,
        })
    return records, checked, violations


def _errors(records, weighted, q):
    out = []
    for r in records:
        pred = _predict(r, weighted, q)
        if pred is None or not r["target"]:
            continue
        out.append((pred - r["target"]) / r["target"])
    return out


def _sweep(records):
    """Tune on the tune split, and only on sales listed inside the buy box — that
    is the band the app makes decisions in."""
    tune = [r for r in records if r["split"] == "tune" and r["target"]
            and r["list_price"] <= BUY_BOX_MAX]
    table = []
    for weighted in (False, True):
        for q in SWEEP:
            errs = _errors(tune, weighted, q)
            if not errs:
                continue
            abs_errs = [abs(e) for e in errs]
            table.append({
                "weighted": weighted, "percentile": q, "n": len(errs),
                "median_abs_pct_error": round(float(np.median(abs_errs)), 4),
                "median_signed_pct_error": round(float(np.median(errs)), 4),
                "within_20pct": round(sum(1 for e in abs_errs if e <= 0.20) / len(abs_errs), 4),
            })
    best = {}
    for weighted in (False, True):
        rows = [r for r in table if r["weighted"] is weighted]
        if rows:
            best["weighted" if weighted else "plain"] = min(
                rows, key=lambda r: r["median_abs_pct_error"]
            )
    return table, best


def wilson(k, n, z=1.96):
    if not n:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(c - h, 4), round(c + h, 4)


def _segment_metrics(records, weighted, q):
    n = len(records)
    covered = sum(1 for r in records if r["comp_count"] >= 3)
    signed = _errors(records, weighted, q)
    abs_errs = [abs(e) for e in signed]
    lo, hi = wilson(sum(1 for e in abs_errs if e <= 0.20), len(abs_errs))
    return {
        "n": n,
        "n_predicted": len(signed),
        "within_20pct_ci": [lo, hi],
        "coverage": round(covered / n, 4) if n else None,
        "median_abs_pct_error": round(float(np.median(abs_errs)), 4) if abs_errs else None,
        "within_10pct": round(sum(1 for e in abs_errs if e <= 0.10) / len(abs_errs), 4)
        if abs_errs else None,
        "within_20pct": round(sum(1 for e in abs_errs if e <= 0.20) / len(abs_errs), 4)
        if abs_errs else None,
        "median_signed_pct_error": round(float(np.median(signed)), 4) if signed else None,
    }


def _ask_benchmark(records):
    """The last asking price used as the prediction. A reference, never a method:
    the asking price can't be a feature (a model that reads it can never find an
    underpriced listing), but it is the bar any model has to clear."""
    out = {}
    for pop, pred in POPULATIONS:
        rows = [r for r in records if r["target"] and pred(r)]
        signed = [(r["list_price"] - r["target"]) / r["target"] for r in rows]
        abs_errs = [abs(e) for e in signed]
        lo, hi = wilson(sum(1 for e in abs_errs if e <= 0.20), len(abs_errs))
        out[pop] = {
            "n": len(rows), "n_predicted": len(signed), "within_20pct_ci": [lo, hi],
            "coverage": None,
            "within_20pct": round(sum(1 for e in abs_errs if e <= 0.20) / len(abs_errs), 4)
            if abs_errs else None,
            "within_10pct": round(sum(1 for e in abs_errs if e <= 0.10) / len(abs_errs), 4)
            if abs_errs else None,
            "median_abs_pct_error": round(float(np.median(abs_errs)), 4) if abs_errs else None,
            "median_signed_pct_error": round(float(np.median(signed)), 4) if signed else None,
        }
    return out


def _segments(records):
    segs = {"overall": {"overall": records}}

    segs["population"] = {name: [r for r in records if r["target"] and pred(r)]
                          for name, pred in POPULATIONS}

    by_price = {name: [] for name, _, _ in PRICE_BANDS}
    for r in records:
        if not r["target"]:
            continue
        b = _band(r["target"], PRICE_BANDS)
        if b:
            by_price[b].append(r)
    segs["price_band"] = by_price

    by_comp = {name: [] for name, _, _ in COMP_BANDS}
    for r in records:
        if r["comp_count"] == 0:
            continue
        b = _band(r["comp_count"], COMP_BANDS)
        if b:
            by_comp[b].append(r)
    segs["comp_band"] = by_comp

    by_zip = {}
    for r in records:
        by_zip.setdefault(r["zip"], []).append(r)
    segs["zip"] = by_zip

    return segs


def _metrics(records, methods):
    by_split = {"tune": [r for r in records if r["split"] == "tune"],
                "test": [r for r in records if r["split"] == "test"]}
    out = {}
    for name, weighted, q in methods:
        out[name] = {}
        for split, split_records in by_split.items():
            segs = _segments(split_records)
            out[name][split] = {
                seg_kind: {sn: _segment_metrics(recs, weighted, q) for sn, recs in seg_group.items()}
                for seg_kind, seg_group in segs.items()
            }
    return out


def _write_errors_csv(path, records, methods):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["url", "address", "zip", "sold_date", "split", "list_price", "close_price",
                    "sqft", "method", "prediction", "pct_error", "comp_count"])
        for r in records:
            for name, weighted, q in methods:
                pred = _predict(r, weighted, q)
                pct = ((pred - r["target"]) / r["target"]
                       if pred is not None and r["target"] else "")
                w.writerow([r["url"], r["address"], r["zip"], r["sold_date"], r["split"],
                            r["list_price"], r["target"] or "", r["sqft"], name,
                            round(pred, 2) if pred is not None else "",
                            round(pct, 4) if pct != "" else "", r["comp_count"]])


def _fmt(v):
    return "n/a" if v is None else f"{v * 100:.1f}%"


def _ci(m):
    lo, hi = m["within_20pct_ci"]
    return "n/a" if lo is None else f"[{lo * 100:.1f}, {hi * 100:.1f}]"


def _row(pop, name, m):
    return (
        f"{pop:<15}{name:<18}{m['n_predicted']:>6}{_fmt(m['within_20pct']):>8}{_ci(m):>16}"
        f"{_fmt(m['within_10pct']):>8}{_fmt(m['median_abs_pct_error']):>10}"
        f"{_fmt(m['median_signed_pct_error']):>13}"
    )


def _headline(metrics, methods, split, ask):
    lines = [
        f"{'population':<15}{'method':<18}{'n':>6}{'w20%':>8}{'95% CI':>16}{'w10%':>8}"
        f"{'med_abs%':>10}{'med_signed%':>13}",
    ]
    for pop, _ in POPULATIONS:
        for name, _, _ in methods:
            lines.append(_row(pop, name, metrics[name][split]["population"][pop]))
        lines.append(_row(pop, "ask (reference)", ask[pop]))
    return lines


def _write_summary(path, metrics, methods, sweep_table, best, checked, violations,
                   coverage, avm_info, ask):
    lines = [
        "FlipFinder ARV backtest summary",
        f"leakage self-check: {checked} targets checked, {violations} violations",
        f"target: close price. {coverage['with_target']} of {coverage['targets']} sales have one "
        f"({coverage['test_with_target']} of {coverage['test']} in test); the rest can't be scored.",
        f"test sales listed at or below {BUY_BOX_MAX}: {coverage['test_primary']} "
        f"({coverage['test_primary_with_target']} with a close price)",
        "",
        "== TEST SPLIT: primary population = sales listed at or below $145k ==",
        *_headline(metrics, methods, "test", ask),
        "",
        f"AVM candidate {avm_info['candidate']}: params {json.dumps(avm_info['params'])}",
        f"  inner validation (tune split only): primary w20 {_fmt(avm_info['inner_primary_w20'])}, "
        f"all w20 {_fmt(avm_info['inner_all_w20'])}, primary n per fold {avm_info['inner_primary_n']}",
        f"  keywords: {json.dumps(avm_info['keywords'])}",
        "",
        f"Percentile sweep (tune split, sales listed at or below {BUY_BOX_MAX} only)",
        f"{'mode':<10}{'pctile':>8}{'n':>7}{'med_abs%':>11}{'med_signed%':>13}{'w20%':>8}",
    ]
    for row in sweep_table:
        mode = "weighted" if row["weighted"] else "plain"
        lines.append(
            f"{mode:<10}{row['percentile']:>8}{row['n']:>7}"
            f"{_fmt(row['median_abs_pct_error']):>11}{_fmt(row['median_signed_pct_error']):>13}"
            f"{_fmt(row['within_20pct']):>8}"
        )
    lines.append("")
    for kind, row in best.items():
        lines.append(f"best {kind}: percentile {row['percentile']} "
                     f"(med_abs {_fmt(row['median_abs_pct_error'])}, "
                     f"signed {_fmt(row['median_signed_pct_error'])})")
    lines.append("")

    for name, _, _ in methods:
        for split in ("tune", "test"):
            lines.append(f"== {name} / {split} ==")
            lines.append(
                f"{'segment':<24}{'n':>6}{'coverage':>10}{'med_abs%':>10}"
                f"{'w10%':>8}{'w20%':>8}{'95% CI':>16}{'med_signed%':>13}"
            )
            for seg_kind, seg_group in metrics[name][split].items():
                for sn, m in seg_group.items():
                    label = sn if seg_kind == "overall" else f"{seg_kind}:{sn}"
                    lines.append(
                        f"{label:<24}{m['n']:>6}{_fmt(m['coverage']):>10}"
                        f"{_fmt(m['median_abs_pct_error']):>10}{_fmt(m['within_10pct']):>8}"
                        f"{_fmt(m['within_20pct']):>8}{_ci(m):>16}"
                        f"{_fmt(m['median_signed_pct_error']):>13}"
                    )
                    if m["n"] < MIN_SAMPLE:
                        lines.append(f"  WARNING: {label} has n={m['n']} (< {MIN_SAMPLE}): low sample size")
            lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _fit_avm(records):
    """Choose the AVM on the tune split only, then predict the test split once."""
    tune = [r for r in records if r["split"] == "tune" and r["target"]]
    test = [r for r in records if r["split"] == "test" and r["target"]]
    ranked = avm.search(tune)
    chosen = ranked[0]
    booster, meta = avm.fit_records(tune, chosen["params"])
    for r in records:
        r["pred"] = {}
    for r, p in zip(test, avm.predict_records(booster, meta, test)):
        r["pred"]["avm"] = p
    info = {"candidate": avm.CANDIDATE, "params": chosen["params"],
            "inner_primary_w20": chosen["inner_primary_w20"],
            "inner_all_w20": chosen["inner_all_w20"],
            "inner_primary_n": chosen["inner_primary_n"],
            "keywords": meta["keywords"],
            "renovated_profile": meta["renovated_profile"]}
    gain = booster.feature_importance(importance_type="gain")
    info["importance"] = sorted(
        ([n, round(float(g), 1)] for n, g in zip(meta["features"], gain)),
        key=lambda x: -x[1])
    return info, ranked


def run(conn, cfg, out_dir=None):
    out_dir = Path(out_dir) if out_dir else (ROOT / "data" / "backtest")
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = _targets(conn)
    records, checked, violations = _build_records(conn, targets)
    sweep_table, best = _sweep(records)
    avm_info, ranked = _fit_avm(records)

    # weighted_median is the current production ARV; tract_median (the plain median
    # of the same comps) is the baseline both have to beat. The AVM is fit on the
    # tune split, so it only has test predictions.
    methods = [("tract_median", False, 0.50), ("weighted_median", True, 0.50),
               ("avm", None, "avm"), ("current_p75", False, 0.75)]
    if "plain" in best:
        methods.append((f"plain_p{int(best['plain']['percentile'] * 100)}",
                        False, best["plain"]["percentile"]))
    if "weighted" in best:
        methods.append((f"weighted_p{int(best['weighted']['percentile'] * 100)}",
                        True, best["weighted"]["percentile"]))

    metrics = _metrics(records, methods)
    test = [r for r in records if r["split"] == "test"]
    coverage = {
        "targets": len(records), "with_target": sum(1 for r in records if r["target"]),
        "test": len(test), "test_with_target": sum(1 for r in test if r["target"]),
        "test_primary": sum(1 for r in test if r["list_price"] <= BUY_BOX_MAX),
        "test_primary_with_target": sum(1 for r in test
                                        if r["list_price"] <= BUY_BOX_MAX and r["target"]),
    }
    ask = _ask_benchmark(test)
    payload = {"sweep": sweep_table, "best": best, "coverage": coverage, "avm": avm_info,
               "avm_search": ranked, "ask_benchmark_test": ask,
               "methods": [{"name": n, "weighted": w, "percentile": q} for n, w, q in methods],
               "metrics": metrics}
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    _write_errors_csv(out_dir / "errors.csv", records, methods)
    _write_summary(out_dir / "summary.txt", metrics, methods, sweep_table, best,
                   checked, violations, coverage, avm_info, ask)

    # every finished candidate's single test score, appended
    log = {"ts": datetime.now().isoformat(timespec="seconds"),
           **{k: v for k, v in avm_info.items() if k != "importance"},
           "test_primary": metrics["avm"]["test"]["population"]["listed<=145k"],
           "test_overall": metrics["avm"]["test"]["population"]["overall"]}
    avm.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(avm.MODEL_DIR / "candidates.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log) + "\n")

    return {
        "n_targets": len(targets),
        "n_tune": sum(1 for r in records if r["split"] == "tune"),
        "n_test": len(test),
        "leakage_checked": checked,
        "leakage_violations": violations,
        "best": best,
        "methods": [n for n, _, _ in methods],
        "primary": {n: metrics[n]["test"]["population"]["listed<=145k"] for n, _, _ in methods[:3]},
    }
