// Redfin blocks non-browser TLS fingerprints (CloudFront WAF), so listing data is
// fetched from a real browser tab instead. Open https://www.redfin.com, paste this
// into DevTools console, then move the downloaded files into FlipFinder/data
// and run: py run.py import
//
// LOCATIONS accepts cities and ZIPs; keep it in sync with config.yaml market.locations.
(async () => {
  const LOCATIONS = ['Dallas, TX', '75080', '75078'];
  const NUM_HOMES = 350;
  const SOLD_WITHIN_DAYS = 180;
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
    const soldCsv = await (await fetch(base + '&sold_within_days=' + SOLD_WITHIN_DAYS)).text();
    save(`active${n}.csv`, activeCsv);
    save(`sold${n}.csv`, soldCsv);
    console.log('  rows:', activeCsv.split('\n').length, soldCsv.split('\n').length);

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
  console.log('done — move active.csv, sold.csv, enrich.json into FlipFinder/data');
})();
