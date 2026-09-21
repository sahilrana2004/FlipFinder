"""Tract assignment (FCC block API) + ACS tract stats + local market stats."""
import time

import numpy as np
import requests

FCC_AREA = "https://geo.fcc.gov/api/census/area"
ACS_BASE = "https://api.census.gov/data/{year}/acs/acs5"
ACS_VARS = "B19013_001E,B25077_001E,B25002_001E,B25002_003E"  # income, value, units, vacant


def _tract_for(lat, lng, cache):
    key = (round(lat, 3), round(lng, 3))
    if key in cache:
        return cache[key]
    try:
        r = requests.get(
            FCC_AREA,
            params={"lat": lat, "lon": lng, "censusYear": 2020, "format": "json"},
            timeout=15,
        )
        results = r.json().get("results", [])
        tract = results[0]["block_fips"][:11] if results else None
    except (requests.RequestException, KeyError, IndexError, ValueError):
        tract = None
    cache[key] = tract
    return tract


def assign_tracts(conn):
    cache = {}
    n = 0
    for table in ("listings", "sold"):
        rows = conn.execute(
            f"SELECT id, lat, lng FROM {table} "
            "WHERE tract IS NULL AND lat IS NOT NULL AND lng IS NOT NULL"
        ).fetchall()
        for row in rows:
            tract = _tract_for(row["lat"], row["lng"], cache)
            if tract:
                conn.execute(f"UPDATE {table} SET tract=? WHERE id=?", (tract, row["id"]))
                conn.commit()  # commit per row: this loop runs minutes, don't starve other writers
                n += 1
            time.sleep(0.15)
    return n


def fetch_acs(conn, cfg):
    year = cfg["census"]["acs_year"]
    tracts = {
        r["tract"]
        for r in conn.execute(
            "SELECT DISTINCT tract FROM listings WHERE tract IS NOT NULL "
            "UNION SELECT DISTINCT tract FROM sold WHERE tract IS NOT NULL"
        )
    }
    have = {r["tract"] for r in conn.execute(
        "SELECT tract FROM area_stats WHERE median_income IS NOT NULL")}
    missing = tracts - have
    counties = {(t[:2], t[2:5]) for t in missing}
    n = 0
    for state, county in counties:
        try:
            r = requests.get(
                ACS_BASE.format(year=year),
                params={"get": ACS_VARS, "for": "tract:*",
                        "in": f"state:{state} county:{county}"},
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
        except (requests.RequestException, ValueError):
            continue
        header, rows = data[0], data[1:]
        idx = {name: i for i, name in enumerate(header)}
        for row in rows:
            tract = row[idx["state"]] + row[idx["county"]] + row[idx["tract"]]
            if tract not in missing:
                continue

            def val(var):
                try:
                    v = float(row[idx[var]])
                    return v if v > 0 else None
                except (TypeError, ValueError):
                    return None

            units, vacant = val("B25002_001E"), val("B25002_003E")
            vacancy = (vacant / units) if units and vacant else None
            conn.execute(
                """INSERT INTO area_stats (tract, median_income, median_home_value, vacancy_rate)
                   VALUES (?,?,?,?)
                   ON CONFLICT(tract) DO UPDATE SET
                     median_income=excluded.median_income,
                     median_home_value=excluded.median_home_value,
                     vacancy_rate=excluded.vacancy_rate,
                     updated=CURRENT_TIMESTAMP""",
                (tract, val("B19013_001E"), val("B25077_001E"), vacancy),
            )
            n += 1
        conn.commit()
    return n


def refresh_market_stats(conn):
    """Per-tract stats computed from our own sold + active data.

    The $/sqft stats are close dollars (sold.close_ppsf). sold.ppsf is the asking
    price per sqft, so a tract median built from it sits above what houses in the
    tract actually close for, and distress's z-score would read every listing as
    cheaper than it is. sold_count therefore counts sales with a recovered close
    price, not every row."""
    tracts = {
        r["tract"]
        for r in conn.execute(
            "SELECT DISTINCT tract FROM listings WHERE tract IS NOT NULL "
            "UNION SELECT DISTINCT tract FROM sold WHERE tract IS NOT NULL"
        )
    }
    for tract in tracts:
        ppsf = [
            r["close_ppsf"]
            for r in conn.execute(
                "SELECT close_ppsf FROM sold WHERE tract=? AND close_ppsf IS NOT NULL",
                (tract,),
            )
        ]
        active = conn.execute(
            "SELECT COUNT(*) c FROM listings WHERE tract=? AND active=1", (tract,)
        ).fetchone()["c"]
        doms = [
            r["dom"]
            for r in conn.execute(
                "SELECT dom FROM listings WHERE tract=? AND active=1 AND dom IS NOT NULL",
                (tract,),
            )
        ]
        med = float(np.median(ppsf)) if ppsf else None
        p75 = float(np.percentile(ppsf, 75)) if ppsf else None
        std = float(np.std(ppsf)) if len(ppsf) >= 2 else None
        med_dom = float(np.median(doms)) if doms else None
        conn.execute(
            """INSERT INTO area_stats
                 (tract, sold_count, active_count, median_sold_ppsf,
                  p75_sold_ppsf, std_sold_ppsf, median_dom)
               VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(tract) DO UPDATE SET
                 sold_count=excluded.sold_count, active_count=excluded.active_count,
                 median_sold_ppsf=excluded.median_sold_ppsf,
                 p75_sold_ppsf=excluded.p75_sold_ppsf,
                 std_sold_ppsf=excluded.std_sold_ppsf,
                 median_dom=excluded.median_dom, updated=CURRENT_TIMESTAMP""",
            (tract, len(ppsf), active, med, p75, std, med_dom),
        )
    conn.commit()
    return len(tracts)
