"""Leakage-safe ARV backtest: for every real sale on disk, predict its price
from comps known as of its own sold_date (production comp selection, so the
backtest can't drift from what the live scorer does), sweep the comp percentile
on the tune split, and score the winners on the untouched test split.

Note what this can and cannot measure: the target is an actual sale price, but
ARV means post-renovation retail. Overpredicting an as-is sale is partly by
design, so treat the sub-$250k bias as an upper bound on true error.
"""
import csv
import json
from pathlib import Path

import numpy as np

from ..config import ROOT
from . import arv as arv_mod

TUNE_FRACTION = 0.7
MIN_SAMPLE = 30
BUY_BOX_MAX = 250000
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


def _band(value, bands):
    for name, lo, hi in bands:
        if (lo is None or value >= lo) and (hi is None or value <= hi):
            return name
    return None


def _predict(rec, weighted, q):
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
        "ORDER BY sold_date"
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

        records.append({
            "url": t["url"], "address": t["address"], "zip": t["zip"],
            "sold_date": t["sold_date"], "price": t["price"], "sqft": t["sqft"],
            "comp_count": len(rows), "split": "tune" if i < n_tune else "test",
            "ppsfs": [r["ppsf"] for r in rows],
            "weights": arv_mod.comp_weights(t, rows, as_of=t["sold_date"]) if rows else [],
        })
    return records, checked, violations


def _errors(records, weighted, q):
    out = []
    for r in records:
        pred = _predict(r, weighted, q)
        if pred is None:
            continue
        out.append((pred - r["price"]) / r["price"])
    return out


def _sweep(records):
    """Tune on the tune split, and only on buy-box-priced sales — that is the
    band the app makes decisions in, and the one the percentile was
    demonstrably miscalibrated for."""
    tune = [r for r in records if r["split"] == "tune" and r["price"] <= BUY_BOX_MAX]
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


def _segment_metrics(records, weighted, q):
    n = len(records)
    covered = sum(1 for r in records if r["comp_count"] >= 3)
    signed = _errors(records, weighted, q)
    abs_errs = [abs(e) for e in signed]
    return {
        "n": n,
        "n_predicted": len(signed),
        "coverage": round(covered / n, 4) if n else None,
        "median_abs_pct_error": round(float(np.median(abs_errs)), 4) if abs_errs else None,
        "within_10pct": round(sum(1 for e in abs_errs if e <= 0.10) / len(abs_errs), 4)
        if abs_errs else None,
        "within_20pct": round(sum(1 for e in abs_errs if e <= 0.20) / len(abs_errs), 4)
        if abs_errs else None,
        "median_signed_pct_error": round(float(np.median(signed)), 4) if signed else None,
    }


def _segments(records):
    segs = {"overall": {"overall": records}}

    by_price = {name: [] for name, _, _ in PRICE_BANDS}
    for r in records:
        b = _band(r["price"], PRICE_BANDS)
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
        w.writerow(["url", "address", "zip", "sold_date", "price", "sqft",
                    "method", "prediction", "pct_error", "comp_count"])
        for r in records:
            for name, weighted, q in methods:
                pred = _predict(r, weighted, q)
                pct = (pred - r["price"]) / r["price"] if pred is not None else ""
                w.writerow([r["url"], r["address"], r["zip"], r["sold_date"],
                            r["price"], r["sqft"], name,
                            round(pred, 2) if pred is not None else "",
                            round(pct, 4) if pct != "" else "", r["comp_count"]])


def _fmt(v):
    return "n/a" if v is None else f"{v * 100:.1f}%"


def _write_summary(path, metrics, methods, sweep_table, best, checked, violations):
    lines = [
        "FlipFinder ARV backtest summary",
        f"leakage self-check: {checked} targets checked, {violations} violations",
        "",
        f"Percentile sweep (tune split, sales at or below {BUY_BOX_MAX} only)",
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
                f"{'segment':<16}{'n':>6}{'coverage':>10}{'med_abs%':>10}"
                f"{'w10%':>8}{'w20%':>8}{'med_signed%':>13}"
            )
            for seg_kind, seg_group in metrics[name][split].items():
                for sn, m in seg_group.items():
                    label = sn if seg_kind == "overall" else f"{seg_kind}:{sn}"
                    lines.append(
                        f"{label:<16}{m['n']:>6}{_fmt(m['coverage']):>10}"
                        f"{_fmt(m['median_abs_pct_error']):>10}{_fmt(m['within_10pct']):>8}"
                        f"{_fmt(m['within_20pct']):>8}{_fmt(m['median_signed_pct_error']):>13}"
                    )
                    if m["n"] < MIN_SAMPLE:
                        lines.append(f"  WARNING: {label} has n={m['n']} (< {MIN_SAMPLE}) — low sample size")
            lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run(conn, cfg, out_dir=None):
    out_dir = Path(out_dir) if out_dir else (ROOT / "data" / "backtest")
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = _targets(conn)
    records, checked, violations = _build_records(conn, targets)
    sweep_table, best = _sweep(records)

    # weighted_median is the production candidate; the two plain methods are the
    # baselines it has to beat.
    methods = [("tract_median", False, 0.50), ("current_p75", False, 0.75),
               ("weighted_median", True, 0.50)]
    if "plain" in best:
        methods.append((f"plain_p{int(best['plain']['percentile'] * 100)}",
                        False, best["plain"]["percentile"]))
    if "weighted" in best:
        methods.append((f"weighted_p{int(best['weighted']['percentile'] * 100)}",
                        True, best["weighted"]["percentile"]))

    metrics = _metrics(records, methods)
    payload = {"sweep": sweep_table, "best": best,
               "methods": [{"name": n, "weighted": w, "percentile": q} for n, w, q in methods],
               "metrics": metrics}
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    _write_errors_csv(out_dir / "errors.csv", records, methods)
    _write_summary(out_dir / "summary.txt", metrics, methods, sweep_table, best,
                   checked, violations)

    return {
        "n_targets": len(targets),
        "n_tune": sum(1 for r in records if r["split"] == "tune"),
        "n_test": sum(1 for r in records if r["split"] == "test"),
        "leakage_checked": checked,
        "leakage_violations": violations,
        "best": best,
        "methods": [n for n, _, _ in methods],
    }
