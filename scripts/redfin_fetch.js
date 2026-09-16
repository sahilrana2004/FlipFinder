// Redfin blocks non-browser TLS fingerprints (CloudFront WAF), so listing data is
// fetched from a real browser tab instead. Open https://www.redfin.com, paste this
// into DevTools console, then move the downloaded files into FlipFinder/data
// and run: py run.py import
//
// LOCATIONS accepts cities and ZIPs; keep it in sync with config.yaml market.locations.
//
// SOLD_ZIPS drives a separate sold-only pull: gis-csv's sold_within_days mixes
// actives into the same 350-row cap as sales, and actives fill it first, so a
// city-wide sold pull mostly returns actives. Querying one ZIP at a time keeps
// each region small enough that sales aren't crowded out. Rebuild this list from
// the unique ZIPs of active listings priced $30,000-$250,000 (plus any ZIP you
// want sold comps for even without current actives there).
(async () => {
  const LOCATIONS = ['Dallas, TX', '75080', '75078'];
  const SOLD_ZIPS = [
    '75043', '75078', '75080', '75150', '75180', '75208', '75210', '75211',
    '75215', '75216', '75217', '75227', '75228', '75232', '75235', '75238',
    '75241', '75249', '75254',
  ];
  const NUM_HOMES = 350;
  const SOLD_WITHIN_DAYS_ZIP = 365;
  const ENRICH_LIMIT = 60;          // detail pages to scrape for remarks + photos
  const PRICE = [30000, 250000];    // buy box, keeps enrichment focused
  const SQFT = [700, 2600];

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

  const targets = [];
  for (const [n, loc] of LOCATIONS.entries()) {
    const region = await lookup(loc);
    console.log('region:', loc, region);
    const base = `/stingray/api/gis-csv?al=1&region_id=${region.id}` +
      `&region_type=${region.type}&uipt=1&sf=1,2,3,5,6,7&num_homes=${NUM_HOMES}&status=9&v=8`;
    const activeCsv = await (await fetch(base)).text();
    save(`active${n}.csv`, activeCsv);
    console.log('  rows:', activeCsv.split('\n').length);

    // pick buy-box rows out of the active CSV to scrape detail pages for
    const lines = activeCsv.split('\n');
    const head = lines[0].split(',').map(h => h.toUpperCase());
    const iUrl = head.findIndex(h => h.includes('URL'));
    const iPrice = head.findIndex(h => h === 'PRICE');
    const iSqft = head.findIndex(h => h.includes('SQUARE FEET') && !h.includes('$'));
    for (const line of lines.slice(1)) {
      const c = line.split(',');
      const price = +c[iPrice], sqft = +c[iSqft], url = (c[iUrl] || '').trim();
      if (!url.startsWith('http')) continue;
      if (price >= PRICE[0] && price <= PRICE[1] && sqft >= SQFT[0] && sqft <= SQFT[1]) {
        targets.push({url, price});
      }
    }
  }

  // run.py import globs sold*.csv, so sold_zip_<zip>.csv is picked up automatically.
  for (const zip of SOLD_ZIPS) {
    const region = await lookup(zip);
    const url = `/stingray/api/gis-csv?al=1&region_id=${region.id}` +
      `&region_type=${region.type}&uipt=1&sf=1,2,3,5,6,7&num_homes=${NUM_HOMES}` +
      `&status=9&v=8&sold_within_days=${SOLD_WITHIN_DAYS_ZIP}`;
    const csvText = await (await fetch(url)).text();
    // Save the raw CSV: `py run.py import` filters to closed sales with a real CSV
    // parser. Splitting on commas here would misread quoted fields and drop sales.
    save(`sold_zip_${zip}.csv`, csvText);
    // Count listing rows by their property URL: the header and Redfin's quoted MLS
    // disclaimer row would otherwise push a capped response to 351 and hide truncation.
    const rowsReturned = (csvText.match(/https:\/\/www\.redfin\.com\/[A-Z]{2}\//g) || []).length;
    const sales = (csvText.match(/,Sold,/g) || []).length;
    const truncated = rowsReturned >= NUM_HOMES ? ' TRUNCATED (sales may be missing)' : '';
    console.log(`${zip} / ${rowsReturned} rows returned / ${sales} sales${truncated}`);
    await new Promise(r => setTimeout(r, 1000));
  }

  targets.sort((a, b) => a.price - b.price);
  const urls = targets.slice(0, ENRICH_LIMIT).map(t => t.url);
  console.log('enriching', urls.length, 'listings...');

  // The full agent write-up is in markup (#marketing-remarks-scroll), not the
  // embedded JSON; the meta description truncates at ~200 chars and cuts off
  // trailing distress keywords ("sold as-is", "bring your vision").
  const meta = /<meta\s+name="description"\s+content="([^"]{40,})"/i;
  const photoRe = /https:\/\/ssl\.cdn-redfin\.com\/photo\/\d+\/[a-z]+photo\/[^"\\\s]+?\.jpg/gi;
  const out = [];
  for (const [i, u] of urls.entries()) {
    try {
      const t = await (await fetch(u)).text();
      const doc = new DOMParser().parseFromString(t, 'text/html');
      const el = doc.querySelector('#marketing-remarks-scroll') ||
                 doc.querySelector('.remarksContainer .remarks');
      let remarks = el ? el.textContent.replace(/\s+/g, ' ').trim() : '';
      if (remarks.length <= 40) {
        const mm = meta.exec(t);
        remarks = mm ? mm[1] : '';
      }
      out.push({url: u, remarks, photos: [...new Set(t.match(photoRe) || [])].slice(0, 6)});
    } catch (e) {
      out.push({url: u, remarks: '', photos: []});
    }
    if (i % 10 === 9) console.log('  ', i + 1, '/', urls.length);
    await new Promise(r => setTimeout(r, 400));
  }
  save('enrich.json', JSON.stringify(out));
  console.log('done — move active*.csv, sold_zip_*.csv, enrich.json into FlipFinder/data');
})();
