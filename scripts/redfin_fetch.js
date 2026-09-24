// Redfin blocks non-browser TLS fingerprints (CloudFront WAF), so listing data is
// fetched from a real browser tab instead. Open https://www.redfin.com, paste this
// into DevTools console, then move the downloaded files into FlipFinder/data
// and run: py run.py import
//
// LOCATIONS accepts cities and ZIPs; keep it in sync with config.yaml market.locations.
//
// SOLD_ZIPS drives a separate sold-only pull: gis-csv's sold_within_days mixes
// actives into the same 350-row cap as sales, and actives fill it first, so a
// city-wide sold pull mostly returns actives. Querying one ZIP at a time, split
// into price bands, keeps each request under the cap. Bands overlap at the edges
// (Redfin filters on a price that isn't always the sold one), so rows are
// de-duplicated by URL. Any band that still returns 350 rows is flagged.
//
// Detail pages: Texas is a non-disclosure state, so the CSV's sold PRICE is the
// last list price, not what the house closed for. The detail page carries the MLS
// fields "RATIO Close Price By Lot Size Ac" and "RATIO List Price By Lot Size Acr";
// close price = list price * close ratio / list ratio. The last list price itself
// comes from the property history (the listing event before the sale, same MLS#).
// Sold detail results are saved in chunks and remembered in localStorage, so a
// re-run after an interruption resumes where it stopped.
(async () => {
  const RUN = {actives: true, sold: true, activeDetail: true, soldDetail: true};
  // The Dallas-wide active pull fills its cap with the whole city's price range, so
  // the ZIPs that hold the bottom of the market are pulled separately.
  const LOCATIONS = ['Dallas, TX', '75080', '75078',
                     '75210', '75215', '75216', '75217', '75241'];
  const SOLD_ZIPS = [
    '75043', '75078', '75080', '75150', '75180', '75208', '75210', '75211',
    '75215', '75216', '75217', '75227', '75228', '75232', '75235', '75238',
    '75241', '75249', '75254',
    // bordering 75210, the one ZIP where >= 20% of sales were at or below $145k
    '75223', '75226',
  ];
  const NUM_HOMES = 350;
  const SOLD_WITHIN_DAYS_ZIP = 1095;   // verified: 730, 1095 and 1825 all honored
  const PRICE_BANDS = [[null, 100000], [100000, 150000], [150000, 200000],
                       [200000, 250000], [250000, 325000], [325000, 425000], [425000, null]];
  // Keep in sync with config.yaml buy_box.max_price: the model and the vision
  // labeler both need remarks and photos for everything the buy box can reach.
  const ACTIVE_DETAIL_MAX = 250000;
  // Every sale, not just the buy box: the close price is only on the detail page,
  // and a model trained on cheap sales alone can't price a renovated house.
  const SQFT = [700, 2600];
  const PACE_MS = 400;
  const CHUNK = 300;
  const DONE_KEY = 'ff_sold_detail_done';

  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const save = (name, text) => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([text]));
    a.download = name;
    a.click();
  };

  // Autocomplete entity types differ from gis-csv region_types. An id sent under
  // the wrong type returns a valid-looking CSV for another place entirely, so only
  // verified mappings are allowed: city -> 6, zip -> 2.
  const AC_TO_GIS = {2: 6, 4: 2};
  const lookup = async (loc) => {
    const ac = await (await fetch('/stingray/do/location-autocomplete?location=' +
      encodeURIComponent(loc) + '&v=2')).text();
    const acJson = JSON.parse(ac.replace('{}&&', ''));
    for (const s of acJson.payload.sections || []) {
      for (const r of s.rows || []) {
        const m = /^(\d+)_(\d+)$/.exec(r.id || '');
        if (!m) continue;
        if (!(+m[1] in AC_TO_GIS)) {
          throw new Error(`${loc} resolved to entity type ${m[1]} — use a city or ZIP`);
        }
        return {type: AC_TO_GIS[+m[1]], id: +m[2], name: r.name};
      }
    }
    throw new Error('no region found for ' + loc);
  };

  // Minimal RFC 4180 parser: remarks-free CSVs, but addresses and the MLS
  // disclaimer row are quoted and can contain commas.
  const parseCsv = (text) => {
    const rows = [];
    let row = [], field = '', q = false;
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      if (q) {
        if (ch === '"' && text[i + 1] === '"') { field += '"'; i++; }
        else if (ch === '"') q = false;
        else field += ch;
      } else if (ch === '"') q = true;
      else if (ch === ',') { row.push(field); field = ''; }
      else if (ch === '\n') { row.push(field.replace(/\r$/, '')); rows.push(row); row = []; field = ''; }
      else field += ch;
    }
    if (field || row.length) { row.push(field); rows.push(row); }
    return rows;
  };
  const csvCell = v => /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v;
  const colIndex = (head, pred) => head.findIndex(h => pred(h.toUpperCase()));

  // Property history is embedded (escaped) JSON; the listing event nearest before
  // the sale, from the same MLS listing, is the last asking price.
  const historyEvents = (text) => {
    const u = text.replace(/\\\\"/g, '"').replace(/\\"/g, '"');
    const k = u.indexOf('"events":[{');
    if (k < 0) return null;
    const start = k + 9;
    let depth = 0, inStr = false;
    for (let j = start; j < u.length; j++) {
      const ch = u[j];
      if (inStr) { if (ch === '\\') j++; else if (ch === '"') inStr = false; continue; }
      if (ch === '"') inStr = true;
      else if (ch === '[' || ch === '{') depth++;
      else if (ch === ']' || ch === '}') {
        depth--;
        if (depth === 0) {
          try { return JSON.parse(u.slice(start, j + 1)); } catch (e) { return null; }
        }
      }
    }
    return null;
  };

  const LIST_EVENT = /^(Listed|Price Changed|Relisted)$/;
  const extractDetail = (text, soldDate) => {
    const doc = new DOMParser().parseFromString(text, 'text/html');
    const el = doc.querySelector('#marketing-remarks-scroll');
    const remarks = el ? el.textContent.replace(/\s+/g, ' ').trim() : '';
    const body = doc.body ? doc.body.textContent.replace(/\s+/g, ' ') : '';
    const num = re => { const m = re.exec(body); return m ? parseFloat(m[1]) : null; };
    const mlsList = num(/(?<!Original |Previous )List Price: ([\d.]+)/);
    const ratioList = num(/RATIO List Price By Lot Size Acr?: ([\d.]+)/);
    const ratioClose = num(/RATIO Close Price By Lot Size Ac: ([\d.]+)/);

    let listPrice = null, listSource = 'no_history';
    const events = historyEvents(text);
    if (events) {
      listSource = 'no_matching_sale';
      const ev = events.filter(e => e.eventDate).sort((a, b) => b.eventDate - a.eventDate);
      const target = soldDate ? Date.parse(soldDate + 'T12:00:00Z') : null;
      const sale = target && ev
        .filter(e => /^Sold/.test(e.eventDescription || '') && Math.abs(e.eventDate - target) < 5 * 864e5)
        .sort((a, b) => /MLS/.test(b.eventDescription) - /MLS/.test(a.eventDescription))[0];
      if (sale) {
        listSource = 'no_listing_event';
        // a listing from before the previous sale belongs to a different transaction
        const prev = ev.find(e => e.eventDate < sale.eventDate - 5 * 864e5 &&
                                  /^Sold/.test(e.eventDescription || ''));
        const valid = ev.filter(e => e.eventDate < sale.eventDate && e.price &&
          LIST_EVENT.test(e.eventDescription || '') && (!prev || e.eventDate > prev.eventDate));
        const same = valid.filter(e => sale.sourceId && e.sourceId === sale.sourceId);
        const pick = (same.length ? same : valid)[0];
        if (pick) {
          listPrice = pick.price;
          listSource = `history:${pick.eventDescription}:${same.length ? 'same_mls' : 'other_mls'}`;
        }
      }
    }
    if (listPrice == null && mlsList) { listPrice = mlsList; listSource = 'mls_field:List Price'; }

    let closePrice = null, closeSource = 'no_ratio_fields';
    if (ratioList && ratioClose && mlsList) {
      closePrice = Math.round(mlsList * ratioClose / ratioList);
      closeSource = 'mls_ratio:close/list*list';
    }
    return {remarks, list_price: listPrice, list_price_source: listSource,
            close_price: closePrice, close_price_source: closeSource,
            mls_list_price: mlsList};
  };

  const activeTargets = [];
  if (RUN.actives) {
    for (const [n, loc] of LOCATIONS.entries()) {
      const region = await lookup(loc);
      console.log('region:', loc, region);
      const base = `/stingray/api/gis-csv?al=1&region_id=${region.id}` +
        `&region_type=${region.type}&uipt=1&sf=1,2,3,5,6,7&num_homes=${NUM_HOMES}&status=9&v=8`;
      const activeCsv = await (await fetch(base)).text();
      save(`active${n}.csv`, activeCsv);
      const rows = parseCsv(activeCsv);
      const head = rows[0];
      const iUrl = colIndex(head, h => h.includes('URL'));
      const iPrice = colIndex(head, h => h === 'PRICE');
      const iSqft = colIndex(head, h => h.includes('SQUARE FEET') && !h.includes('$'));
      console.log('  rows:', rows.length - 1);
      for (const c of rows.slice(1)) {
        const price = +c[iPrice], sqft = +c[iSqft], url = (c[iUrl] || '').trim();
        if (!url.startsWith('http')) continue;
        if (price > 0 && price <= ACTIVE_DETAIL_MAX && sqft >= SQFT[0] && sqft <= SQFT[1]) {
          activeTargets.push({url, price});
        }
      }
    }
  }

  const soldTargets = [];
  if (RUN.sold) {
    // run.py import globs sold*.csv, so sold_zip_<zip>.csv is picked up automatically.
    for (const zip of SOLD_ZIPS) {
      const region = await lookup(zip);
      let head = null;
      const byUrl = new Map();
      const flags = [];
      for (const [lo, hi] of PRICE_BANDS) {
        let url = `/stingray/api/gis-csv?al=1&region_id=${region.id}` +
          `&region_type=${region.type}&uipt=1&sf=1,2,3,5,6,7&num_homes=${NUM_HOMES}` +
          `&status=9&v=8&sold_within_days=${SOLD_WITHIN_DAYS_ZIP}`;
        if (lo) url += `&min_price=${lo}`;
        if (hi) url += `&max_price=${hi}`;
        const rows = parseCsv(await (await fetch(url)).text());
        head = head || rows[0];
        const iUrl = colIndex(head, h => h.includes('URL'));
        let n = 0;
        for (const r of rows.slice(1)) {
          const u = (r[iUrl] || '').trim();
          if (!u.startsWith('https://www.redfin.com/')) continue;   // MLS disclaimer row
          n++;
          byUrl.set(u, r);
        }
        if (n >= NUM_HOMES) flags.push(`${lo || 0}-${hi || 'max'}`);
        await sleep(800);
      }
      const iStatus = colIndex(head, h => h === 'STATUS');
      const iDate = colIndex(head, h => h === 'SOLD DATE');
      const iPrice = colIndex(head, h => h === 'PRICE');
      const iUrl = colIndex(head, h => h.includes('URL'));
      const rows = [...byUrl.values()];
      const sold = rows.filter(r => (r[iStatus] || '').toLowerCase() === 'sold');
      const dates = sold.map(r => new Date(r[iDate].replace(/-/g, ' '))).filter(d => !isNaN(d)).sort((a, b) => a - b);
      save(`sold_zip_${zip}.csv`, [head, ...rows].map(r => r.map(csvCell).join(',')).join('\n'));
      const truncated = flags.length ? ` TRUNCATED bands ${flags.join(', ')} (sales may be missing)` : '';
      console.log(`${zip} / ${rows.length} rows / ${sold.length} sales / ` +
        `${dates.length ? dates[0].toISOString().slice(0, 10) + '..' + dates.at(-1).toISOString().slice(0, 10) : '-'}` +
        ` / ${sold.filter(r => +r[iPrice] <= 145000).length} at or below $145k${truncated}`);
      for (const r of sold) {
        const d = new Date(r[iDate].replace(/-/g, ' '));
        if (!isNaN(d)) {
          const iso = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
          soldTargets.push({url: r[iUrl].trim(), sold_date: iso});
        }
      }
    }
  }

  if (RUN.activeDetail && activeTargets.length) {
    // The full agent write-up is in markup (#marketing-remarks-scroll), not the
    // embedded JSON; the meta description truncates at ~200 chars and cuts off
    // trailing distress keywords ("sold as-is", "bring your vision").
    const meta = /<meta\s+name="description"\s+content="([^"]{40,})"/i;
    const photoRe = /https:\/\/ssl\.cdn-redfin\.com\/photo\/\d+\/[a-z]+photo\/[^"\\\s]+?\.jpg/gi;
    const out = [];
    console.log('enriching', activeTargets.length, 'active listings...');
    for (const [i, t] of activeTargets.entries()) {
      try {
        const text = await (await fetch(t.url)).text();
        const doc = new DOMParser().parseFromString(text, 'text/html');
        const el = doc.querySelector('#marketing-remarks-scroll') ||
                   doc.querySelector('.remarksContainer .remarks');
        let remarks = el ? el.textContent.replace(/\s+/g, ' ').trim() : '';
        if (remarks.length <= 40) {
          const mm = meta.exec(text);
          remarks = mm ? mm[1] : '';
        }
        out.push({url: t.url, remarks, photos: [...new Set(text.match(photoRe) || [])].slice(0, 6)});
      } catch (e) {
        out.push({url: t.url, remarks: '', photos: []});
      }
      if (i % 10 === 9) console.log('  ', i + 1, '/', activeTargets.length);
      await sleep(PACE_MS);
    }
    save('enrich.json', JSON.stringify(out));
  }

  if (RUN.soldDetail && soldTargets.length) {
    let done = new Set();
    try { done = new Set(JSON.parse(localStorage.getItem(DONE_KEY) || '[]')); } catch (e) {}
    const todo = [...new Map(soldTargets.map(t => [t.url, t])).values()].filter(t => !done.has(t.url));
    console.log(`sold detail: ${todo.length} to fetch, ${done.size} already done`);
    const stamp = Date.now();
    let buf = [], part = 0;
    const flush = () => {
      if (!buf.length) return;
      save(`sold_detail_${stamp}_${part++}.json`, JSON.stringify(buf));
      buf.forEach(b => done.add(b.url));
      try { localStorage.setItem(DONE_KEY, JSON.stringify([...done])); } catch (e) {}
      buf = [];
    };
    for (const [i, t] of todo.entries()) {
      try {
        const res = await fetch(t.url);
        if (res.ok) buf.push({url: t.url, sold_date: t.sold_date, ...extractDetail(await res.text(), t.sold_date)});
        else console.log('  HTTP', res.status, t.url);
      } catch (e) {
        console.log('  failed', t.url, e.message);
      }
      if (buf.length >= CHUNK) flush();
      if (i % 50 === 49) console.log('  ', i + 1, '/', todo.length);
      await sleep(PACE_MS);
    }
    flush();
  }
  console.log('done — move active*.csv, sold_zip_*.csv, enrich.json, sold_detail_*.json into FlipFinder/data');
})();
