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
| `streamlit run app.py` | Start the application. |
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

## 5. Language model and privacy

| `SANKET_LLM_PROVIDER` | Where data goes | Notes |
|---|---|---|
| `offline` (default) | Nowhere | Plans are assembled by a deterministic rule engine that quotes the records. |
| `ollama` | Your Ollama server | Local inference. Needs Ollama running and a model pulled (for example `ollama pull llama3.1:8b`, about 8 GB RAM; a larger model follows the structured-output schema more reliably). |
| `anthropic` | Anthropic API | Default model `claude-opus-5-5`, via the official SDK's structured outputs. |
| `openai` | OpenAI or a compatible endpoint | Set `SANKET_LLM_BASE_URL` for a self-hosted compatible server. |

- External providers are **blocked** until `SANKET_ALLOW_EXTERNAL_LLM=true` is set. When enabled, only
  the extracted records of the *selected mine* are sent, per plan request. The sidebar always states the
  current data flow.
- `SANKET_REDACT_PERSON_NAMES=true` (default) removes victims' names from what is sent to a model.
- Check your provider's retention and processing terms before enabling it for official data.
- API keys are read from the environment variable named in `SANKET_LLM_API_KEY_ENV` (default
  `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`). No key is stored in the code or the repository.
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

`python -m pytest tests -q` - **65 tests, all passing** (about 30 s). They cover:

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

---

## 7. Known limitations

- **The LLM path has not been run against a live model.** No API key or local model was available
  during development. The adapter, prompts, schema and validator are tested with simulated model
  output; the Anthropic, OpenAI and Ollama calls themselves are untested. The offline rule engine is
  the tested default.
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

1. Run the LLM path against the approved provider or a local model and review plans with mining
   safety experts before operational use.
2. Link mines to an authoritative register (DGMS mine codes, coordinates, operator) to confirm identity
   across eras and enable region and map views.
3. Add verified context sources: approved emergency plans, DGMS circulars and regulations, geological
   and hydrological reports, with provenance.
4. Ingest further sources (DGMS annual reports, court-of-inquiry reports, earlier SANKET volumes) for
   narratives of historic disasters and for non-fatal dangerous occurrences.
5. Add authentication, access logging and a review/sign-off workflow for generated plans.
6. Have a domain expert review the alias list and the "likely duplicates" list in the audit.
