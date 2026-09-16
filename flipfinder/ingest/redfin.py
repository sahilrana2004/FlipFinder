"""Redfin unofficial stingray endpoints: region autocomplete + gis-csv search."""
import csv
import io
import json
import re
from datetime import datetime

import requests

UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
AUTOCOMPLETE = "https://www.redfin.com/stingray/do/location-autocomplete"
GIS_CSV = "https://www.redfin.com/stingray/api/gis-csv"


# Autocomplete entity types and gis-csv region_types use different numbering.
# Passing an id under the wrong region_type returns a valid-looking CSV for a
# completely different place (id 32148 is Richardson TX as type 2, Chestertown NY
# as type 6), so unmapped types must raise rather than fall through.
AC_TYPE_TO_GIS = {2: 6, 4: 2}  # city -> 6, zip -> 2; both verified live


def lookup_region(query):
    r = requests.get(AUTOCOMPLETE, params={"location": query, "v": 2}, headers=UA, timeout=30)
    r.raise_for_status()
    payload = json.loads(r.text.replace("{}&&", "", 1))
    for section in payload.get("payload", {}).get("sections", []):
        for row in section.get("rows", []):
            m = re.fullmatch(r"(\d+)_(\d+)", row.get("id", ""))
            if m:
                ac_type = int(m.group(1))
                if ac_type not in AC_TYPE_TO_GIS:
                    raise ValueError(
                        f"{query!r} resolved to Redfin entity type {ac_type}, which has "
                        "no verified gis-csv region_type. Use a city or ZIP."
                    )
                return AC_TYPE_TO_GIS[ac_type], int(m.group(2)), row.get("name", query)
    raise ValueError(f"No Redfin region found for {query!r}")


def fetch_csv(region_type, region_id, num_homes, sold_within_days=None):
    params = {
        "al": 1,
        "region_id": region_id,
        "region_type": region_type,
        "uipt": "1",              # single-family houses only
        "sf": "1,2,3,5,6,7",
        "num_homes": num_homes,
        "status": 9,
        "v": 8,
    }
    if sold_within_days:
        params["sold_within_days"] = sold_within_days
    r = requests.get(GIS_CSV, params=params, headers=UA, timeout=60)
    r.raise_for_status()
    text = r.text
    if text.lstrip().startswith("{"):
        raise RuntimeError(f"Redfin returned an error payload: {text[:200]}")
    return list(csv.DictReader(io.StringIO(text)))


def _col(row, *fragments):
    for key, val in row.items():
        k = (key or "").upper()
        if all(f in k for f in fragments):
            return val
    return None


def _num(v):
    if v is None:
        return None
    v = str(v).replace(",", "").replace("$", "").strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _parse_common(row):
    sqft = _num(_col(row, "SQUARE FEET"))
    price = _num(_col(row, "PRICE"))
    return {
        "address": (_col(row, "ADDRESS") or "").strip(),
        "city": (_col(row, "CITY") or "").strip(),
        "state": (_col(row, "STATE") or "").strip(),
        "zip": (_col(row, "ZIP") or "").strip(),
        "price": price,
        "beds": _num(_col(row, "BEDS")),
        "baths": _num(_col(row, "BATHS")),
        "sqft": sqft,
        "lot_sqft": _num(_col(row, "LOT SIZE")),
        "year_built": _num(_col(row, "YEAR BUILT")),
        "dom": _num(_col(row, "DAYS ON MARKET")),
        "ppsf": (price / sqft) if price and sqft else None,
        "lat": _num(_col(row, "LATITUDE")),
        "lng": _num(_col(row, "LONGITUDE")),
        "url": (_col(row, "URL") or "").strip(),
        "mls": (_col(row, "MLS") or "").strip(),
        "status": (_col(row, "STATUS") or "").strip(),
        "sold_date": (_col(row, "SOLD DATE") or "").strip(),
    }


def deactivate_all(conn):
    """Call once before importing every region — the active flag is rebuilt from
    the full sweep, so per-region deactivation would drop earlier regions."""
    conn.execute("UPDATE listings SET active=0")
    conn.commit()


def upsert_active_rows(conn, rows):
    n = 0
    for row in rows:
        d = _parse_common(row)
        if not d["url"] or not d["price"]:
            continue
        conn.execute(
            """INSERT INTO listings
               (mls, url, address, city, state, zip, price, beds, baths, sqft,
                lot_sqft, year_built, dom, ppsf, lat, lng, status, active)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)
               ON CONFLICT(url) DO UPDATE SET
                 price=excluded.price, dom=excluded.dom, ppsf=excluded.ppsf,
                 status=excluded.status, active=1, last_seen=CURRENT_TIMESTAMP""",
            (d["mls"], d["url"], d["address"], d["city"], d["state"], d["zip"],
             d["price"], d["beds"], d["baths"], d["sqft"], d["lot_sqft"],
             d["year_built"], d["dom"], d["ppsf"], d["lat"], d["lng"], d["status"]),
        )
        n += 1
    conn.commit()
    return n


def _parse_sold_date(raw):
    """Redfin sold dates come as 'April-15-2026'; store ISO so SQL date
    comparisons (used by the point-in-time ARV backtest) work directly."""
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%B-%d-%Y").date().isoformat()
    except ValueError:
        return None


def upsert_sold_rows(conn, rows):
    """gis-csv sold_within_days returns active listings and closed sales mixed
    together (actives fill the 350-row cap first), so anything not an actual
    closed sale must be dropped here rather than trusted from the endpoint."""
    kept = skipped = 0
    for row in rows:
        d = _parse_common(row)
        if not d["url"] or not d["price"]:
            skipped += 1
            continue
        if d["status"].strip().lower() != "sold":
            skipped += 1
            continue
        sold_date = _parse_sold_date(d["sold_date"])
        if not sold_date:
            skipped += 1
            continue
        conn.execute(
            """INSERT INTO sold
               (url, address, city, zip, price, sold_date, beds, baths, sqft,
                year_built, ppsf, lat, lng)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(url) DO UPDATE SET
                 price=excluded.price, sold_date=excluded.sold_date""",
            (d["url"], d["address"], d["city"], d["zip"], d["price"], sold_date,
             d["beds"], d["baths"], d["sqft"], d["year_built"], d["ppsf"],
             d["lat"], d["lng"]),
        )
        kept += 1
    conn.commit()
    return kept, skipped


def ingest_listings(conn, cfg):
    m = cfg["market"]
    deactivate_all(conn)
    results = []
    for loc in m["locations"]:
        rtype, rid, name = lookup_region(loc)
        rows = fetch_csv(rtype, rid, m["max_listings"])
        results.append((name, upsert_active_rows(conn, rows)))
    return results


def ingest_solds(conn, cfg):
    m = cfg["market"]
    results = []
    for loc in m["locations"]:
        rtype, rid, name = lookup_region(loc)
        rows = fetch_csv(rtype, rid, m["max_listings"],
                         sold_within_days=m["sold_within_days"])
        results.append((name, upsert_sold_rows(conn, rows)))
    return results
