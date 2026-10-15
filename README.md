# SANKET AI

**Historical Mine Inundation Intelligence & Emergency Decision Support**

SANKET AI helps mining authorities in India (District Magistrates, mine managers, safety officers) who
take charge of mines in an unfamiliar region. For a selected mine it shows the water-related accidents
recorded in the DGMS SANKET volumes, what the DGMS inquiry found would have prevented them, and an
evidence-grounded emergency response plan. Every statement links to the source PDF page.

It is a decision-support aid. It does not replace approved emergency plans, real-time monitoring,
professional engineering judgment or official orders.

---

## 1. What the source documents actually contain

The two supplied PDFs are the DGMS statistical volumes *SANKET - Statistical Analysis & Numbers: Key
Evaluation of Trends* for **coal mines** (`sanket0404_2024.pdf`, 316 pages) and **non-coal mines**
(`DOC-20260831-WA0001..pdf`, 355 pages). They are general accident-statistics volumes, not inundation
registers. The schema and the application were derived from what is in them:

| Section in each volume | What it gives per record | Used as |
|---|---|---|
| Summary of findings of fatal accidents, 2016-2022 | date, time, mine, owner, district, state, persons killed, DGMS narrative, DGMS findings ("Had ... been ..., this accident could have been averted"), DGMS cause code | incidents with narrative and preventive findings |
| Major accidents since 1901 (4+ deaths) | date, mine name, killed, seriously injured, cause | historic incidents (table entry only) |
| Courts of inquiry | date, mine name, cause, killed, court and assessors | merged into the matching historic incident |

Consequences that shape the application:

- **Water-related records are a subset.** A record is shown as water-related when DGMS classifies the
  cause as *Irruption of Water* / *Inundation* or *Drowning in Water*, or when the narrative of an
  accident classified otherwise mentions water (labelled separately as a keyword match).
- **"Preventive measures" are DGMS findings of precautions that were *not* in place.** The volumes do
  not record measures implemented afterwards, their dates or their effectiveness. The UI labels them
  accordingly and marks present status as unknown.
- **Not in the source at all:** geology or hydrogeology, the DGMS region of an individual mine, water
  level as a measured field, narratives before 2016, non-fatal inundations at a named mine, and any
  present-day condition. These are shown as "Not documented in source", never filled in.
- **Historic tables print only a mine name.** Such mines have no state, district or owner and are
  identified by name only; they are never merged automatically with a located mine of the same name.

The full extraction audit is generated at `data/processed/extraction_audit.md` (machine-readable:
`extraction_report.json`) and is also shown in the app under *Data audit and system status*.

Current extraction result: 1,063 raw records -> 1,006 incidents at 769 mines (433 coal, 336 non-coal);
73 water-related incidents at 68 mines; 1,537 DGMS findings. The number of fatal-accident records
parsed matches the total each volume states for every year 2016-2022.

---

## 2. Quick start

Requires Python 3.11+ (developed on 3.12, Windows 11). No GPU is needed.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env                # optional; defaults work offline

python -m src.ingestion               # extract, normalise, validate, index
streamlit run app.py                  # open http://127.0.0.1:8501
```

The app listens on IPv4 loopback only. If `http://localhost:8501` is slow to connect or fails, use
`http://127.0.0.1:8501`. When starting it from a script or other non-interactive shell, add
`--server.headless true` so Streamlit's first-run email prompt does not block startup.

Place the two SANKET PDFs in the project folder or in `data/raw/` (set `SANKET_PDF_DIRS` to use other
folders). File names do not matter: each volume is recognised from its title pages. The PDFs are opened
read-only and are never moved or modified.

### Commands

| Command | Purpose |
|---|---|
| `python -m src.ingestion` | Build the database and indexes. Does nothing if the PDFs, alias file and pipeline version are unchanged. |
| `python -m src.ingestion --rebuild` | Force a full rebuild (use after the source documents change). |
| `python -m src.ingestion --skip-vectors` | Skip the semantic index (keyword retrieval only). |
| `streamlit run app.py` | Start the application (the chatbot is the **Ask SANKET** button on every page). |
| `uvicorn src.api:app --host 127.0.0.1 --port 8600` | Optional chat HTTP API with streaming (needs `SANKET_API_TOKEN`). |
| `python -m pytest tests -q` | Run the automated tests. |
| `python -m src.evaluation` | Run the evaluation suite on the real dataset. |

If the PDFs are missing or unreadable, ingestion stops with exit code 2 and says why. It never
substitutes sample data, and the app shows recovery instructions instead of placeholder mines.

The first ingestion downloads the embedding model (all-MiniLM-L6-v2, about 90 MB) once. If that fails
(for example on an air-gapped machine), ingestion still completes and retrieval uses keywords only;
rerun `python -m src.ingestion` later to add the semantic index.

---

## 3. Using the application

1. **Home** - choose *Coal Mines* or *Metal / Non-Coal Mines*. Counts are computed from the records.
2. **Mine directory** - search by mine name, district, state or owner; filter by state and district;
   pick from the type-ahead selector or the table. Similar names are told apart by location and owner.
   By default only mines with water-related records are listed; a toggle shows every mine.
3. **Mine dashboard**
   - **A. Overview** - identity, location, counts of documented records, data gaps, and any
     same-named record groups that could not be confirmed as this mine.
   - **B. Inundation timeline** - chart plus one expandable card per incident with the verbatim DGMS
     narrative and findings, and the source file, pages and extracted text.
   - **C. Preventive findings** - each DGMS finding with its accident, statutory provisions cited,
     citation, and "present status: unknown".
   - **D. Emergency response plan** - six stages; each step states its basis (this mine's records or
     general practice), the evidence, responsible role, priority, what to verify and what approvals are
     needed, followed by missing information, assumptions, the validation report and the agent trace.
     Download as Markdown or JSON.
   - **Search this mine's records** - keyword + semantic search restricted to the selected mine.
4. **Document chat** - free-form questions about anything in the two volumes. Retrieval
   covers every text page (1,280 passages: statistics, tables, findings, historic lists) plus the
   structured records of any mine named in the question. The model must take at least 80% of the
   answer from the passages, with a citation marker `[S#]` per sentence. At most 20% may be general
   background knowledge, each such sentence marked and shown as *(general knowledge)*. Every answer is
   checked before it is shown (see section 4a), and the result is shown with the cited passages. With
   no usable model, the chat shows the best-matching passages instead. It is opened from the floating
   chatbot (item 5); there is no separate chat page. Code: `src/chat.py`.
5. **Floating chatbot** - an **Ask SANKET** button in the bottom-right corner of every page opens the
   same chat in a panel over the page (a Streamlit fragment, so the page underneath is not reloaded).
   It provides streaming drafts, a verification status per answer, expandable sources, conversation
   history (open, rename, delete, clear), memory controls, **Stop** and **Retry**. It works on desktop
   and mobile widths. Code: `src/chat_ui.py`.

Labels used throughout: *Documented in source* (blue), *Recommendation / Inference* (amber),
*Not documented in source* (grey).

---

## 4. Architecture

```
PDFs --> pdf_extractor --> data_normalizer --> SQLite (source of truth) --> retrievers --> agent --> UI
                                          \--> FTS5 keyword index + Chroma vector index (ranking only)
```

```
app.py                       Home and category routing
pages/1_Mine_Directory.py    Searchable directory
pages/2_Mine_Intelligence.py Mine dashboard (overview, timeline, findings, plan, search)
pages/3_Data_Audit.py        Extraction audit and system status
src/config.py                Environment-driven settings
src/pdf_extractor.py         PyMuPDF parsers for the three record structures
src/data_normalizer.py       Mine identity, dates, water classification, findings, duplicates
src/schemas.py               Pydantic models (records, tool I/O, plan output)
src/database.py              SQLite storage and exact structured queries
src/vector_store.py          Local Chroma index with on-device embeddings
src/ingestion.py             Pipeline, audit report, CLI
src/retrievers.py            Tools 1-5, hybrid ranking, evidence bundle
src/prompts.py               System and task prompts
src/llm.py                   Provider adapter (offline / ollama / anthropic / openai)
src/agent.py                 Tool orchestration, offline rule engine, plan assembly
src/safety_validation.py     Checks every draft against the evidence
src/citations.py             Citations, evidence serialisation, PDF verification
src/report.py                Markdown export
src/chat.py                  Document chat: page retrieval, water-record index, 80/20 grounding prompt
src/verification.py          Claim-level checking of a drafted answer against the passages
src/chat_service.py          Chat orchestration: typed event stream (status/token/citation/done/error)
src/chat_store.py            Conversations, messages and memory notes (SQLite, per-user isolation)
src/chat_cache.py            Retrieval/answer cache (SQLite or Redis), keyed on the document version
src/chat_ui.py               Floating chatbot (bottom-right on every page)
src/api.py                   HTTP API with Server-Sent Events (FastAPI)
src/evaluation.py            Evaluation suite
src/ui.py                    Shared Streamlit helpers
config/mine_aliases.json     Curated spelling variants
tests/                       Automated tests
```

### Data schema (derived from the documents)

- **mines** - `mine_id`, name and spelling variants, category, state, district, operator (all nullable),
  `identity_basis` (`name_state_district` or `name_only`), record counts, possibly-related mine ids.
- **incidents** - `incident_id`, `mine_id`, date as printed, ISO date (null when ambiguous), year,
  `date_quality`, DGMS cause code and label, `water_relevance` and its stated basis, narrative,
  findings block, casualty counts, quantities mentioned in the narrative, provisions cited, court of inquiry.
- **preventive findings** - `measure_id`, `mine_id`, `incident_id`, verbatim text, keyword category
  (nullable), provisions cited, `implementation_date` = null, `reported_outcome` = null, `current_status` = unknown.
- **source references** - `source_id`, file name, pages, section, verbatim extracted text, extraction
  method, verification status and warnings.

Stable ids: incidents such as `coal-2017-F054` (volume, year, serial printed in the source),
`coal-MAJ-217` (table serial); mines are a slug plus a hash of (category, name, state, district).

### Agent workflow

A fixed sequence of LangChain tools, not an open prompt:

1. `mine_directory_search` - exact SQL filters, fuzzy matching only for discovery.
2. `mine_record_retriever` - canonical record by `mine_id`.
3. `inundation_history_retriever` - incidents filtered by exact `mine_id`, then optional year/type.
4. `preventive_measures_retriever` - findings for that mine and those incidents.
5. `context_and_guidance_retriever` - provisions cited in the mine's own records, plus verified external
   context if supplied; otherwise reports the limitation.
6. `emergency_plan_generator` - evidence bundle -> draft (LLM or offline rule engine) -> Pydantic
   validation -> safety validation -> final plan.

The generator writes only the draft. Mine identity, the incident list, findings and citations in the
final output are assembled from the database. The safety validator then removes or repairs: citations
not in the mine's bundle, evidence-labelled steps without a source, figures and dates not found in the
evidence, other mines' names, probabilities / return periods / predictions, claims that a historical
measure is in place today, and order-like wording. Everything it changes is listed in the plan's
validation report. With no water-related record, the output is a clearly labelled generic framework.

Retrieval is hybrid: exact `mine_id` lookup decides membership; BM25 (SQLite FTS5) and vector
similarity are fused by reciprocal rank only to order passages inside that mine.

---

## 4a. Document chat: verified answers, memory and cache

```
question -> validate + rate limit -> store -> resolve follow-up into a standalone query (model call)
         -> retrieve: pages of both PDFs (BM25 + vectors, fused) + records of named mines
                      + for water questions, the complete extracted list of water-related records
         -> draft answer, streamed as "token" events (shown greyed as an unverified draft)
         -> verify claims -> redraft once with feedback if too little survives
         -> "citation" events for the sources the final answer uses -> store -> "done"
```

**Web facts.** With `SANKET_CHAT_WEB=true`, each question is also searched on DuckDuckGo (`src/web_search.py`,
no API key; only the question is sent) in parallel with the PDF search. Up to 4 results become passages
`W1..W4`. The model is asked to add context the PDFs lack (causes, background, aftermath), cite it `[W#]`,
and never let it override the PDFs. Web facts plus general knowledge are capped at 20% of the answer by
word count; extra outside sentences are removed automatically. When the PDFs do not answer, web facts are
shown under an explicit "from internet sources (not from the SANKET documents)" heading. Results are
cached for a day.

**Checking modes** (`SANKET_CHAT_VERIFY_MODE`). `fast` (default): the answer streams normally, then the
deterministic checks below run instantly (citations exist; numbers occur in the cited source) and failing
sentences are removed. The label is "Citations checked", which says honestly that meaning was not checked
by a second model. `thorough`: adds the independent model check and one redraft (labels "Sources verified"
/ "Partly verified"), at the cost of 5-30 s per answer. `off`: no checks ("Not checked"). Live timings
with Groq: about 5-8 s per answer in fast mode, against 20-60 s in thorough mode.

**Verification** (`src/verification.py`, thorough mode; its deterministic part also runs in fast mode). The draft is split into claims (sentences; citations at the
end of a paragraph cover its earlier sentences). Each claim is typed by its marker: documented `[S#]`,
inference `[INF]`, general knowledge `[GK]`, a statement that the documents lack the answer, or
unmarked. Documented, inferred and unmarked claims are checked twice:

1. **Deterministic:** citation markers must refer to retrieved passages, and every number of two or more
   digits must occur in the supporting passage. A four-digit year also matches the volumes'
   `dd.mm.yy` dates.
2. **Independent model check:** a separate call, given only the relevant passages, rates each claim
   supported / partial / unsupported / contradicted. It is told that a figure from another row is a
   contradiction, and that claims of absence or completeness need explicit support.

A claim is kept only if both checks pass. Inferences may be partially supported, and an unmarked claim
is kept only if the checker names a supporting passage, which then becomes its citation. Citation
markers in the final text are rebuilt from validated ids, so an invented marker never reaches the user.
General knowledge is not checked against the PDFs; it is capped at 20% and labelled.

The answer status comes from these results, not from the model's own confidence:

| Status | Rule |
|---|---|
| verified | Nothing was removed. |
| partially verified | At least half of the documented content survived. |
| insufficient evidence | Nothing documented survived, or the documents lack the answer. |
| unverified | The checker could not run; only deterministic checks were applied. Labelled as such. |

If less than half survives, the answer is redrafted once with the rejected claims as feedback and more
passages.

**Water-record index.** Questions that mention water, inundation, irruption or drowning also receive one
passage listing every water-related record extracted from both PDFs. Each row carries its page, and the
header gives exact totals per class. This was added after live testing: without it, counting and listing
questions saw only the best-matching pages and gave wrong totals. With it, live answers matched the
database (coal 38 irruption + 12 drowning; non-coal 4 + 4).

**Memory layers.**

- *Active:* the last three answered exchanges are sent with each question (long answers are trimmed).
  Follow-ups are rewritten into a standalone search query.
- *Persistent:* conversations and messages are stored in `data/chat/chat.sqlite3`, with role, content,
  status, citations, check details and timestamps. Each user is identified by an unguessable token
  (`?u=...` in the app, the `X-Sanket-User` header in the API) and stored as its SHA-256. Every query
  filters by owner, so another user's conversation id returns "not found".
- *Long-term:* per-conversation topic notes built from the user's own questions only (never from model
  claims). They are used only when the user refers back ("earlier", "you said", "remind me"), are
  passed as "context only - not evidence", and can be viewed, deleted or switched off.

**Cache** (`src/chat_cache.py`). Retrieval results and verified answers are cached in a local SQLite
file, or Redis if `SANKET_REDIS_URL` is set. Every key includes the document version (ingestion
fingerprint of both PDFs + pipeline version + retrieval-logic version), so changing a PDF invalidates
everything; old entries are purged when the cache opens. Answers are cached only for questions asked
without prior conversation context. A cached answer is reused only after its passages are confirmed to
still exist in the current index.

**HTTP API** (`src/api.py`, `uvicorn src.api:app`). `POST /api/chat/stream` returns Server-Sent Events
(`status`, `token` = unverified draft, `citation`, `done`, `error`). Conversation CRUD and memory endpoints
are also provided. The API requires `SANKET_API_TOKEN` (it refuses all requests when unset) plus a
per-user token, and validates request bodies. See [HOW_TO_RUN.md](HOW_TO_RUN.md) section 8 for the
contract.

**Limits and safety.** Questions are limited per user per minute (`SANKET_CHAT_RATE_LIMIT_PER_MIN`) and
in length. Answers are rendered as Markdown without HTML, and source excerpts as plain text. Retrieved
text is delimited and treated as data. Conversation text is not written to the logs.

---

## 5. Language model and privacy

| `SANKET_LLM_PROVIDER` | Where data goes | Notes |
|---|---|---|
| `offline` (default) | Nowhere | Plans are assembled by a deterministic rule engine that quotes the records. |
| `groq` | Groq Cloud | Default model `openai/gpt-oss-120b` (key `GROQ_API_KEY`). **Configured on this machine.** |
| `openrouter` | OpenRouter | Default model `nvidia/nemotron-3-super-120b-a12b:free` (key `OPENROUTER_API_KEY`). **Configured here as the fallback.** |
| `ollama` | Your Ollama server | Local inference. Needs Ollama running and a model pulled (for example `ollama pull llama3.1:8b`, about 8 GB RAM; a larger model follows the structured-output schema more reliably). |
| `anthropic` | Anthropic API | Default model `claude-opus-5-5`, via the official SDK's structured outputs. |
| `openai` | OpenAI or a compatible endpoint | Set `SANKET_LLM_BASE_URL` for a self-hosted compatible server. |

- External providers are **blocked** until `SANKET_ALLOW_EXTERNAL_LLM=true` is set. When enabled, only
  the extracted records of the *selected mine* are sent, per plan request, and for the chat only the
  passages retrieved for each question. The sidebar always states the current data flow.
- `SANKET_LLM_FALLBACK_PROVIDERS` lists providers to try, in order, when the primary one fails before
  producing output. This machine uses Groq first and OpenRouter as the fallback. Each answer records
  which provider wrote it.
- `SANKET_REDACT_PERSON_NAMES=true` (default) removes victims' names from what is sent to a model.
- Check your provider's retention and processing terms before enabling it for official data.
- API keys are read from the environment variable named in `SANKET_LLM_API_KEY_ENV` (default
  `GROQ_API_KEY` / `OPENROUTER_API_KEY` / `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`). No key is stored in
  the code or the repository; `.env` and `data/chat/` are git-ignored.
- Embeddings are computed locally. No document text is sent anywhere for indexing.
- Text extracted from PDFs is treated as untrusted data: it is delimited, angle brackets are neutralised
  and the system prompt instructs the model to ignore embedded instructions. The model has no tools and
  cannot take actions.
- Streamlit is configured to listen on `localhost` only (`.streamlit/config.toml`). Add authentication
  before exposing it on a network.

### Adding verified external context (optional)

Put JSON files in `data/context/`. Each entry needs `title`, `text`, `provenance`, `retrieved_on` (ISO
date) and `mine_ids` (a list of ids, or `"all"`). Entries without provenance or date are ignored and
reported. External context is shown and passed to the generator separately from the SANKET records.

### Spelling variants

`config/mine_aliases.json` maps a printed variant to the canonical printed name within a category.
Three entries are included, each documented in the file. Look-alike names detected automatically are
listed in the audit for review and are not merged.

---

## 6. Tests and evaluation

`python -m pytest tests -q` - **116 tests, all passing** (about 30 s; no API calls). Besides the areas
below, `tests/test_chat.py` covers page indexing, chat retrieval, grounding measurement, prompt rules,
provider fallback, blocked external calls and error messages. `tests/test_chat_service.py` covers:

- per-user isolation and conversation CRUD, and memory controls;
- cache invalidation when the document version changes;
- claim typing and splitting;
- removal of unsupported claims and invented figures, and rebuilding of wrong citation markers;
- refusals reported as insufficient evidence, and the 20% general-knowledge cap;
- the "unverified" label when the checker is down, and redrafting;
- event order, where the stored answer is the verified one and not the draft;
- follow-up resolution, and memory used only as context;
- cache reuse, passages-only mode, validation, rate limits and interrupted generation;
- the SSE API (auth, streaming, isolation);
- the floating chatbot (open, ask, close, reopen, history).

Those tests replace only the language model with scripted fakes; retrieval, verification logic, storage
and UI are real. The other test files cover:

- **Ingestion**: both PDFs processed; parsed counts equal the volumes' stated yearly totals; page
  references valid; table cells kept together; missing and corrupted PDFs reported; no sample-data
  fallback; incremental rerun; source PDFs unmodified; duplicates, aliases, nulls and ambiguous dates.
- **Directory**: category filters; search by name, location and structured filters; similar names
  disambiguated; clear empty state.
- **Retrieval**: for all 769 mines, every returned record carries the requested `mine_id`;
  same-named mines share nothing; sources and pages retained; no fictional incidents; hybrid search
  cannot leave the mine; external context requires provenance.
- **Agent**: schema conformance; evidence use; findings vs recommendations; missing geology disclosed;
  generic framework without evidence; fabricated model output (invented year, figure, citation, other
  mine, probability, "now installed", order wording) is removed; model failure falls back; prompt
  injection text stays inside the evidence block; external providers blocked without opt-in.
- **Citations**: format, redaction, and verification that cited text is on the cited PDF pages.
- **UI** (Streamlit AppTest): category selection, directory filtering and search, mine selection,
  dashboard content per mine, plan generation, empty, stale and missing-database states.

Synthetic fixtures use obviously artificial names (`TESTMINE ...`), live in temp folders and never
enter `data/processed`.

`python -m src.evaluation` on the real dataset (68 mines with water-related records):

| Metric | Result |
|---|---|
| Mine-identity accuracy | 513 / 513 records and citations (100%) |
| Citation correctness (excerpt found on cited PDF pages) | 90 / 90 (100%) |
| Retrieval relevance, top-1 (14 queries on multi-record mines) | keyword 100%, hybrid 100% |
| Plans passing validation (offline generator) | 68 / 68; 912 steps, 232 backed by the mine's records |
| Unsupported statements removed by the validator | 0 of 912 (offline generator) |

The relevance sample is small because few mines have several narrated records; treat it as a smoke
test, not a benchmark.

**Live chat checks (Groq with OpenRouter fallback, real PDFs)**, run during development:

| Question | Result |
|---|---|
| What happened at Sayal D colliery in 2017 and what could have prevented it? | Verified; correct narrative and DGMS finding, cited to pp. 118-119 |
| Which regulation did that finding cite? (follow-up) | Verified; Regulations 129(1) and 127(1) & (3), CMR 1957 |
| Fatal irruption/drowning accidents, coal vs non-coal | Verified; 38 + 12 coal and 4 + 4 non-coal, equal to the database |
| Which of those happened in Jharkhand? (follow-up) | Verified; Sayal D, Khoodia, Moonidih (coal), Turamdih, Purnadih Nawada (non-coal) |
| What did DGMS find could have prevented the Kuardih drowning? | Verified; both findings, cited to p. 119 |
| Present groundwater level at Turamdih / water level at Chasnalla | "Not established from the documents" |
| Same question asked twice | Second answer served from cache in 0.1 s |
| Groq rate-limited or request too large | OpenRouter answered automatically |
| HTTP API stream | `status -> token -> status -> citation -> done`; a second user got 404 for the first user's conversation |

Before the water-record index was added, live testing produced wrong totals and a wrong date that the
checker did not catch. This is why that index exists, and why automated checking is described here as
reducing, not eliminating, errors.

---

## 7. Known limitations

- **Live model coverage is partial.** The document chat has been run end to end against Groq and the
  OpenRouter fallback, and through the HTTP API. Emergency plans written by a language model (instead of
  the rule engine) have not been reviewed live. The Anthropic, OpenAI and Ollama calls are untested.
- **Chat verification reduces but cannot eliminate errors.** The checker is itself a language model.
  Deterministic checks cover citations and numbers, not names: a misspelt mine name in an otherwise
  supported sentence ("Turamddh") was not caught. Wrong pairings within a long table are possible when
  the passage is ambiguous. Always open the cited page before acting.
- **Free-tier model limits.** Groq's free tier rejects requests above roughly 8k tokens per minute; the
  OpenRouter fallback then answers, more slowly (often 30 s to 2 min). Paid tiers avoid this.
- **Chat identity is a private link, not a login.** Anyone with the `?u=...` link (or the API user token)
  can read that user's conversations. Add real authentication before multi-user deployment.
- **Rate limiting is per process.** Running several app or API processes multiplies the limit; use a
  shared limiter (for example on Redis) in production.
- **Stopping generation** interrupts the app's processing, but the provider may finish generating the
  draft in the background (and bill for it).
- **Refusal fallback is not enabled** for the Anthropic provider: if the model declines, the offline
  plan is shown with a note, rather than rerouting official data to another model automatically.
- **Coverage** is limited to what the two volumes print: narratives only for fatal accidents in
  2016-2022; earlier history only for accidents with four or more deaths; no geology, monitoring data
  or present-day status. A mine without a record is not thereby a mine without inundation history.
- **Mine identity across eras is unresolved by design.** For example the historic "Chasnalla" entries
  (1975, 1976) and the located "CHASNALLA" (Dhanbad, 2016 record) are separate entries linked as
  "possibly related". Confirming such links needs an authoritative mine register.
- **Dates in the non-coal historic table** use two-digit years and, for some rows, an ambiguous
  day/month order. The century is inferred from table order and flagged; ambiguous rows keep only the year.
- **Source inconsistencies are preserved, not corrected** (for example a district printed with a state
  that looks wrong, or owner names spelt differently between years).
- **Water-keyword records** may involve water only incidentally; they are labelled as keyword matches.
- **Finding categories** are assigned by keyword rules for navigation and may be wrong or empty.
- **The safety validator is rule-based.** It catches unsupported figures, dates, names, citations and
  set phrases; it cannot judge every qualitative claim a language model might make. Human review of
  every plan remains necessary.
- **No authentication, audit log or role-based access.** The app is single-user and local.
- **OCR is not implemented** because every record-bearing page has a text layer. A scanned volume would
  be reported as a blocker.

## 8. Recommended next steps

1. Review chat answers and language-model plans with mining safety experts before operational use.
   Build a fixed question set with expected answers to measure the chat's accuracy over time.
2. Add a deterministic check of mine names and places against the passages, and replace the private-link
   identity with real authentication if the app is shared.
2. Link mines to an authoritative register (DGMS mine codes, coordinates, operator) to confirm identity
   across eras and enable region and map views.
3. Add verified context sources: approved emergency plans, DGMS circulars and regulations, geological
   and hydrological reports, with provenance.
4. Ingest further sources (DGMS annual reports, court-of-inquiry reports, earlier SANKET volumes) for
   narratives of historic disasters and for non-fatal dangerous occurrences.
5. Add authentication, access logging and a review/sign-off workflow for generated plans.
6. Have a domain expert review the alias list and the "likely duplicates" list in the audit.
