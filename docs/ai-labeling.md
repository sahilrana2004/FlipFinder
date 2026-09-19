# AI labeling

A local vision model looks at each scored listing's photos, remarks, and numbers, and records what
it sees: condition, renovation scope, red flags, and whether the remarks contradict the photos.
The verdict (pass / maybe / deal) is not asked of the model — it is derived from the listing's
current margin. The map is where you browse the results.

Human labels in `labels` remain the only training data. `fitter.py` reads `labels` and nothing
else; AI output lives in `ai_labels` and never crosses over.

## Commands

```
py run.py refresh      # import -> census -> score -> ailabel -> score -> refresh verdicts
py run.py ailabel      # label what is unlabeled, re-score, refresh verdicts
py run.py serve        # UI at http://127.0.0.1:8321
```

`refresh` is the normal entry point, and `pipeline` is an alias of it. Re-running it with no new
data makes zero model calls: the model is contacted only when a listing is eligible.

A listing is eligible when it is in `scores`, has at least one photo, and has no `ai_labels` row
for the current model and prompt version. Labeling is resumable — an interrupted run picks up
where it stopped. Listings without photos are skipped and counted.

Config lives in `config.yaml` under `ollama`:

```yaml
ollama:
  url: "http://localhost:11434"
  model: "qwen3.5:9b"
  photos_per_listing: 4
```

Requests pass `think: false`, `format: "json"`, and `num_ctx: 16384`. All three matter: the model
is a thinking model whose reasoning text otherwise breaks JSON parsing, and four full-size
`bigphoto` images plus the prompt overflow Ollama's default 4096-token context with an HTTP 400.

Photos come from `ssl.cdn-redfin.com` with the user agent in `ingest/redfin.py`. Never request
redfin.com pages or APIs from Python; its WAF blocks non-browser clients, which is why listing and
sold data arrive as browser-downloaded CSVs.

## How a verdict is produced

1. The model returns `notes`, `condition` (1-10, photos only), `reno_scope` (light / medium / gut,
   photos and remarks together), `red_flags`, `photos_representative`, and `text_conflict`.
2. `photos.downgrade_reasons()` derives the downgrade in code. It fires when a red flag matches
   one of `MATERIAL_FLAGS` — foundation/structural, water damage/leak/flood, fire/smoke, mold,
   roof, unfinished or incomplete work, tenant occupied — or when `text_conflict` is true. An
   as-is sale on its own is not a downgrade; it is a sale term, not a defect. The matched flags
   become the reasons shown in the UI.
3. `photos.margin_ceiling(margin)` maps the margin to the highest allowed verdict: below 0% pass,
   0-10% maybe, above 10% deal.
4. `photos.ai_verdict(margin, downgrade)` returns `max(0, ceiling - downgrade)`.

The verdict is computed from the margin at read time, so it cannot go stale when a re-score moves
the margin. `photos.refresh_verdicts()` also writes the derived value into
`ai_labels.suggested_verdict` after every re-score, because the Metrics agreement query reads that
column.

### Why the model does not decide the verdict

An earlier version asked for an absolute verdict. 28 of 33 answers simply restated the margin
ceiling, and verdicts went stale as soon as the model's own condition changed the reno tier.
Asking only for a downgrade decision failed differently: across four prompt revisions the same
pilot listings flipped between versions, downgrading on renovation depth (already priced in),
on rooms that were not photographed, or on nothing concrete. Deriving it from evidence-bound
flags is reproducible and auditable, at the cost of inheriting whatever noise is in the flags.

## Reno scope drives renovation cost

`arv._reno_tier()` uses `listings.reno_scope` when it is set, falling back to the old
condition/distress logic. Photos flatter; the scope is judged from the remarks too, so it is the
better basis for cost. A human-corrected condition (`condition_source = 'human'`) still wins, so
that fixing a condition in Triage keeps affecting reno cost.

The tier selects a cost per sqft from `reno_cost_per_sqft` (light 25, medium 50, gut 90), which
changes renovation cost, margin, and score. ARV is never affected.

## Schema

`ai_labels` — one row per (listing, model, prompt version):

| column | meaning |
| --- | --- |
| `condition` | 1-10, what the photos show |
| `reno_scope` | light / medium / gut |
| `red_flags` | JSON array, model's own wording |
| `photos_representative` | 0/1 |
| `text_conflict` | 0/1, remarks contradict the photos |
| `downgrade` | 0/1, derived in code |
| `suggested_verdict` | derived; refreshed after each re-score |
| `reasons` | JSON array, the matched flags |
| `notes`, `photo_count`, `seconds`, `raw_response`, `ts` | provenance |

Added to `listings`: `condition_score`, `condition_source` (`'ai'` / `'human'` / NULL),
`reno_scope`. Added to `labels`: `ai_label_id`, `ai_shown`, `condition_human`.

New columns are applied by an idempotent migration in `init_db()` that checks `PRAGMA table_info`
and issues `ALTER TABLE ... ADD COLUMN` only for what is missing (`CREATE TABLE IF NOT EXISTS`
never alters an existing table). Running `init_db` twice is safe.

## UI

**Map** — pins are colored by AI verdict: deal green, maybe amber, pass muted red, unlabeled grey,
with a legend. The filter above the map (All / Maybe + Deal / Deal only) persists in
`localStorage` inside try/catch. Ranking still comes from the scorer, so a high-scoring listing
can be an AI pass; use the filter, not the rank, to find the AI's picks.

The listing panel shows the AI block: condition, reno scope, red flag chips, notes, an amber
warning on `text_conflict`, and the verdict as `Margin allows X` or
`Margin allows X - AI downgraded to Y because ...`. Model text is HTML-escaped, since it echoes
listing remarks.

**Triage and Metrics** still work and are optional. Triage orders AI-labeled listings first, and
flagged ones (text conflict or any red flag) ahead of those. Blind mode hides the algo score and
the AI block, and the condition select is not prefilled, so the suggestion cannot leak. Every
human label records `ai_label_id` and `ai_shown`, and `/api/metrics` reports verdict agreement
overall and split by shown vs blind, a 3x3 confusion matrix, and condition corrections — so
anchoring is measurable if manual labeling resumes.

## API

`GET /api/listings` — adds `ai_verdict` (derived), `condition`, `reno_scope`, `red_flag_count`,
`text_conflict`.

`GET /api/listing/{id}` and `GET /api/triage/next` — include `ai_label` for the current model and
prompt version, or null. The payload carries `downgrade`, `margin_ceiling`, and a
`suggested_verdict` derived at read time.

`POST /api/labels` — human labels. Stores `ai_label_id`, `ai_shown` (1 when the suggestion was
visible), and `condition_human`. A condition override also sets `listings.condition_score` and
`condition_source = 'human'`. On a re-label `ai_shown` only ratchets up: once seen, the verdict
cannot be treated as blind.

## Refreshing listing data

Redfin data arrives as browser downloads. Put `active0.csv`, `active1.csv`, `active2.csv`, and
`enrich.json` into `data/` (archive the old ones under `data/archive/<date>/` rather than deleting
them), then run `py run.py refresh`.

`import` deactivates every listing first and re-activates whatever appears in the new files, so
listings missing from a pull drop out. Two things to know before reading that as "sold":

- **The export caps at 350 rows per region.** Dallas and Prosper hit the cap on every pull so far,
  so those files are a window on the market, not its inventory. A listing absent from a new file
  may simply have fallen outside the window. Per-ZIP pulls, like the existing `sold_zip_*.csv`
  files, would remove the ambiguity.
- `enrich.json` applies remarks and photos only to the URLs it contains. Remarks already stored
  for other listings are kept, so enrichment coverage carries over between pulls.

## Known issues

- **ACS tract stats have never loaded.** `api.census.gov` returns an HTML "Missing Key" page with
  HTTP 200, and `census.fetch_acs` swallows it as a JSON error, so `area_stats.median_income`,
  `median_home_value`, and `vacancy_rate` are NULL for every tract. The API needs a key.
- **Red flag noise reaches verdicts.** The prompt forbids speculative flags, but the model still
  produces some ("older roof on exterior photo", "unverified structural elements"). Since the
  downgrade is derived from flags, that noise becomes verdict noise. It currently only bites where
  a margin allows maybe or deal.
- **Nothing verifies the labels.** Labels are applied automatically; `labels` stays empty unless
  someone uses Triage.
- **Model answers vary between runs** (no temperature is set). Stored labels are stable because
  `refresh` never re-asks, but bumping `PROMPT_VERSION` re-rolls every listing.
- **Condition is photos-only by design**, which understates houses whose remarks describe deep
  work. `reno_scope` is the counterweight, and it is what sets cost.
- **Speed depends on VRAM.** Labeling ran at ~15 s per listing with the model fully on the GPU and
  ~58 s when part of it spilled to the CPU.

## What changed on the `ai-prelabeling` branch

| file | why |
| --- | --- |
| `config.yaml` | `ollama.model` pointed at `qwen2.5vl:7b`, which is not installed — that is why no listing ever got a condition. Now `qwen3.5:9b`, 4 photos per listing. |
| `flipfinder/db.py` | `ai_labels` table; idempotent migration for the new `listings`, `labels`, and `ai_labels` columns. |
| `flipfinder/analysis/photos.py` | Replaced the condition-only photo scorer with the labeler: prompt, validation with one retry, failure log, resumable eligibility, derived downgrade and verdict, verdict refresh. |
| `flipfinder/analysis/arv.py` | `_reno_tier` prefers `listings.reno_scope`. |
| `flipfinder/analysis/metrics.py` | `ai_agreement`: verdict agreement overall and split by shown vs blind, confusion matrix, condition corrections. |
| `flipfinder/app/server.py` | AI fields on the listings, listing, and triage endpoints; `POST /api/labels` records anchoring and condition overrides; triage orders flagged listings first. |
| `flipfinder/app/static/index.html` | Map filter, verdict pins and legend, AI block with the verdict line, condition select in the label widget, AI agreement section on Metrics. |
| `run.py` | `ailabel` and `refresh` commands; `pipeline` is an alias of `refresh`; `enrich` no longer runs photo scoring, since labeling needs scores that do not exist at that point. |

Run reports and raw outputs from the runs that produced the current labels are under
`data/refresh/`, `data/ailabel/`, and `data/autolabel/` (all gitignored): input file tables,
per-version pilot outputs, batch logs, failure logs, and before/after top 15 comparisons.

## Failures

Bad responses are retried once, then logged to `data/autolabel/failures.log` with the listing and
the raw response, and the run continues. The listing stays eligible, so the next run retries it.
Validation rejects a non-integer or out-of-range condition, an unknown reno scope, a non-list
where a list is required, and non-string notes — except that a list of strings is joined, which
the model does often enough to matter.
