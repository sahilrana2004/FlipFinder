# The web UI

`flipfinder/app/static/index.html` is the whole front end: one file, no build step,
no framework. This document covers how it is put together and why, so the next
change to it doesn't have to rediscover the constraints.

## What the UI is for

The pipeline scores listings and a local vision model labels them; the app is where
those results get browsed and, optionally, corrected. Three views:

- **Map** — a ranked, filterable list beside a map of every scored listing, and a
  detail panel for the one you picked.
- **Triage** — one unlabeled listing at a time, keyboard-driven, for recording your
  own verdict.
- **Metrics** — whether the scorer and the model agree with the labels you gave.

Human labels stay the only ground truth: the UI never writes model output into
`labels`, and the Metrics view keeps AI agreement separate from scorer quality.

## Layout

Everything floats over a full-bleed map. The top bar, list panel, detail panel, map
controls and legend are translucent panels; Triage and Metrics are full pages that
cover the map behind a scrim. That keeps one visual model — chrome over a map —
instead of three unrelated screens.

Panel geometry is driven by two custom properties on `body`: `--li` (left inset,
list panel width or 0 when collapsed) and `--ri` (right inset, detail panel width or
0 when closed). The legend, map controls, attribution and every `panInside` /
`fitBounds` call read those, so nothing ends up hidden under a panel.

Breakpoints: 1180px narrows the panels, 980px stacks Triage and Metrics into one
column, 760px turns the list into a bottom sheet and the detail panel into a
full-width sheet.

## Design system

Tokens live in `:root` and are re-declared under `:root[data-theme="light"]`. An
inline script in `<head>` picks the theme (stored choice, else OS preference) before
first paint, so there is no flash; the toggle persists a choice, and the OS setting
is followed until one is made.

**Glass** is three cheap layers, not a shader: `backdrop-filter: blur() saturate()`
for the material, a gradient hairline rim lit from the top-left (a masked
pseudo-element), and an inner bevel. All three are static, so panning the map only
re-blurs — nothing re-rasterizes. `@supports` and `prefers-reduced-transparency`
both fall back to a solid surface; `prefers-reduced-motion` collapses transitions.

**Verdict colors are status colors**, not decoration, and they are validated rather
than picked by eye:

| | deal | maybe | pass | worst CVD pair | normal vision | contrast |
|---|---|---|---|---|---|---|
| dark (on `#15181e`) | `#54c398` | `#f09c17` | `#c95368` | ΔE 12.2 | ΔE 20.9 | all ≥ 3:1 |
| light (on `#f7f8fa`) | `#19966e` | `#c8800d` | `#b32035` | ΔE 8.5 | ΔE 20.2 | all ≥ 3:1 |

ΔE is OKLab ×100 under simulated protanopia/deuteranopia, all pairs (a map is a
scatter plot: any two pins can be neighbours). Pin **radius** repeats the verdict —
deal 7.5, maybe 6.5, pass and unlabeled 5 — so the map does not depend on hue, and
every chip carries its label. Costs in the deal-math bar are greys; only the spread
takes a verdict color.

Type is the system stack (`Segoe UI Variable Text` / `Display` first): no webfont
request, correct hinting on Windows. Money and metrics use `tabular-nums`; hero
figures use proportional figures.

## Rendering and state

`S` holds rows, the id index, the derived `shown` list, and the current
filter/sort/query/selection. `refresh()` recomputes `shown`, re-renders the list,
syncs markers, and optionally refits the map. Everything else is a pure function of
that state.

- **Routing** is the hash: `#/map`, `#/map/<id>`, `#/triage`, `#/metrics`. A listing
  has a URL, Back closes the panel, and keyboard navigation uses `replaceState` so
  arrowing through the list doesn't fill history.
- **Markers** render into a single canvas (`preferCanvas`, `L.canvas`), and hover,
  click and tooltips are handled once on a `featureGroup` rather than per marker.
  Draw order is sorted by verdict so deals sit above passes where pins overlap.
- **The list renders in chunks of 120**, extended by an `IntersectionObserver` as
  you scroll. Cards also carry `content-visibility: auto`, so off-screen rows in the
  rendered chunk skip layout and paint.
- **Detail payloads** are cached in a 24-entry LRU keyed by listing id, warmed by a
  120ms hover dwell on a row, so clicking a row usually renders instantly. The cache
  entry is dropped when that listing is labeled.
- **Escaping**: everything from the API — addresses, remarks, model notes, tags —
  goes through `esc()` before it reaches `innerHTML`, and the tooltip is built with
  `textContent`. Listing and photo URLs must parse as https on `redfin.com` or
  `cdn-redfin.com` or they are not rendered.

## Assets and third parties

**Leaflet 1.9.4** loads from jsDelivr with SRI hashes. The stylesheet sits in the
body, after the shell markup, so the shell paints without waiting for the CDN while
still being guaranteed to apply before the deferred scripts run. If Leaflet fails to
load, the map area says so and the list keeps working.

**Tiles: Stadia Alidade Smooth** (`alidade_smooth`, `alidade_smooth_dark`).

- The previous OpenStreetMap tiles were inverted with a CSS filter per tile to fake
  a dark map; a real dark style is cheaper and looks better.
- CARTO's basemaps now stamp "API KEY REQUIRED" across keyless tiles.
- Stadia serves `localhost` / `127.0.0.1` without an API key, rate limited, which is
  exactly how this app is served. It **401s without a Referer**, so never add
  `referrerpolicy="no-referrer"` to the page or the tile layer.
- Retina tiles are taken only at `devicePixelRatio >= 2` (`r` option overrides
  Leaflet's own `> 1` test). At 1.5x they cost roughly 35MB of image memory for
  labels that are barely sharper.
- Leaflet's stylesheet loads after the inline `<style>`, so map overrides are scoped
  under `#map` to win on specificity.

**Photos**: Redfin keeps every photo at two sizes, and either stored form maps to
both — `…/mbphoto/<dir>/genMid.<file>.jpg` is 623×414 (~41KB) and
`…/bigphoto/<dir>/<file>.jpg` is 1280×960 (~600KB). Most rows in `photos` store the
1280px original. The UI derives both and asks for the size it actually displays:
thumbnails and the detail gallery take 623px, Triage's larger gallery takes 1280px
through `srcset`, and any derived URL that 404s falls back to the stored one.

## Performance

Measured in headless Chrome against the same server and data, old UI vs new, at 35
listings and on a synthetic 2,000-listing database:

| | old · 35 | new · 35 | old · 2,000 | new · 2,000 |
|---|---|---|---|---|
| first rows/pins rendered | 45ms | 39ms | 199ms | 135ms |
| load event | 43ms | 25ms | 44ms | 20ms |
| DOM nodes | 196 | 1,837 | 2,161 | 4,989 |
| SVG marker nodes | 35 | 0 | 2,000 | 0 |
| event listeners | 233 | 154 | 4,163 | 170 |
| JS heap after GC | 1.3MB | 1.4MB | 5.6MB | 4.6MB |
| tile image memory | 7MB | 6MB | 7MB | 7MB |
| verdict filter switch | 0.6ms | 2.1ms | 6.4ms | 16.8ms |

Thumbnails: 19 visible rows decode to 17.7MB using the 623px variant, against
66.1MB if the stored URL is used as-is.

The filter switch at 2,000 listings rebuilds the canvas layer set and the first
chunk of rows in one frame. If that ever matters, diff the marker set instead of
clearing and re-adding it.

## Accessibility

Semantic controls throughout: the list is a `listbox` with `aria-activedescendant`,
the verdict filter is a `radiogroup` with arrow-key support, toggles carry
`aria-pressed`, icon-only buttons carry labels, and the photo viewer is a modal
dialog that returns focus. Panels that are off-screen or behind another view are
`inert`, so Tab never lands in them. Focus rings are visible on every control.

## Known limits

- Stadia's keyless tiles are rate limited; sustained heavy panning could start
  returning 429s. There is no API-key option in `config.yaml` yet.
- The list grows as you scroll and never releases rows; with thousands of listings
  and a long scroll session, the DOM keeps what you have seen.
- `/api/listings` is fetched once at load. After `py run.py refresh`, reload the page
  to see new scores.
- Blind mode hides the score, deal math and AI block in the panel and in Triage, but
  map pins are still colored by verdict.
