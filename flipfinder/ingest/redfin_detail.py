"""Enrich listings with remarks + photo URLs scraped from the Redfin detail page.

The gis-csv export has no listing description or photos, so we pull each detail
page and regex the embedded JSON. Polite by default: rate-limited, capped per run.
"""
import html as html_mod
import re
import time

import requests

from .redfin import UA

# The agent's full write-up lives in markup, not the embedded JSON. The meta
# description only carries its first ~200 chars, which truncates away distress
# keywords ("sold as-is", "bring your vision") that often trail the blurb.
REMARKS_DIV = re.compile(
    r'<div class="remarks"[^>]*id="marketing-remarks-scroll".*?>(.*?)</div>', re.S
)
TAGS = re.compile(r"<[^>]+>")
META_DESC = re.compile(r'<meta\s+name="description"\s+content="([^"]{40,})"', re.I)
PHOTO_RE = re.compile(
    r'https://ssl\.cdn-redfin\.com/photo/\d+/[a-z]+photo/[^"\\\s]+?\.jpg', re.I
)


def _extract_remarks(html):
    m = REMARKS_DIV.search(html)
    if m:
        text = html_mod.unescape(TAGS.sub("", m.group(1)))
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 40:
            return text
    m = META_DESC.search(html)
    return html_mod.unescape(m.group(1)) if m else None


def enrich(conn, cfg):
    e = cfg["enrich"]
    rows = conn.execute(
        """SELECT id, url FROM listings
           WHERE active=1 AND remarks IS NULL AND url LIKE 'http%'
           ORDER BY last_seen DESC LIMIT ?""",
        (e["max_detail_fetches"],),
    ).fetchall()
    done = failed = 0
    for row in rows:
        try:
            r = requests.get(row["url"], headers=UA, timeout=30)
            if r.status_code != 200:
                failed += 1
                continue
            html = r.text
            remarks = _extract_remarks(html) or ""
            conn.execute("UPDATE listings SET remarks=? WHERE id=?", (remarks, row["id"]))
            for url in dict.fromkeys(PHOTO_RE.findall(html)[:8]):
                conn.execute(
                    "INSERT OR IGNORE INTO photos (listing_id, url) VALUES (?,?)",
                    (row["id"], url),
                )
            conn.commit()
            done += 1
        except requests.RequestException:
            failed += 1
        time.sleep(e["delay_seconds"])
    return done, failed


def import_sold_details(conn, items):
    """Apply sold detail-page results (scripts/redfin_fetch.js, sold_detail_*.json).
    A later scrape of the same sale overwrites an earlier one."""
    n = 0
    for item in items:
        cur = conn.execute(
            """UPDATE sold SET list_price=?, list_price_source=?, close_price=?,
                 close_price_source=?, remarks=?, detail_fetched=CURRENT_TIMESTAMP
               WHERE url=?""",
            (item.get("list_price"), item.get("list_price_source"), item.get("close_price"),
             item.get("close_price_source"), item.get("remarks") or None, item["url"]),
        )
        n += cur.rowcount
    conn.commit()
    return n
