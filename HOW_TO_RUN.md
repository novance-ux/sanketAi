# Inundation: How to Run It

This guide covers what Inundation does, what to set up before the first run, how to start it, and how to use it.
For architecture, data schema and test details, see [README.md](README.md).

---

## 1. What it does

Inundation is a local web application for mining officials (District Magistrates, mine managers, safety
officers). It is meant for officers taking charge of mines in a region they do not know.

You choose a mine, and the application shows:

1. **Its recorded history of water-related accidents.** These are inundations, irruptions of water and
   drownings, taken from the two DGMS SANKET PDF volumes (coal and non-coal mines).
2. **What the DGMS inquiry found.** For each accident, these are the precautions that DGMS said would have
   prevented it ("Had ... been done, this accident could have been averted"). They describe what was
   *missing at the time*. They are not proof that anything was installed afterwards.
3. **A step-by-step emergency response plan** in six stages, from detecting the problem through to recovery.
   Each step says whether it is based on that mine's own records or on general safety practice.

4. **A document chat** for asking any question about the two volumes. Answers come from the PDFs with
   page citations; at most 20% may be clearly labelled general background knowledge.
   The chat is available from a **floating chat button** in the bottom-right corner of every page.
   Every statement in an answer is **checked against the PDF passages before the answer is
   shown**; unsupported statements are removed. Conversations are saved so you can reopen them later.

Every fact links to the PDF file and page it came from, so you can check it before acting.

**What it cannot tell you:** geology, current water levels, present mine conditions, or whether a
precaution is in place today. The PDFs do not contain this information. A mine with no record here may
still have flooded; it only means no record was found in these two volumes.

> The plan is a decision-support aid. It is not an official order. It does not replace the mine's approved
> emergency plan or the judgement of qualified engineers.

---

## 2. What you need

| Item | Detail |
|---|---|
| Computer | Windows 10/11 (also works on Linux/macOS). About 2 GB free disk space. No GPU needed. |
| Python | Version 3.11 or newer. Check with `python --version`. |
| Internet | Needed once to install packages and download a 90 MB search model. After that it runs offline. The chat (and language-model plans) need internet to reach the model provider. |
| The two PDFs | The SANKET coal volume and the SANKET non-coal volume. |

The PDFs are already in this folder (`sanket0404_2024.pdf` and `DOC-20260831-WA0001..pdf`). File names
do not matter, because the program identifies each volume from its title pages.

---

## 3. Before the first run (one-time setup)

Open **PowerShell** in the project folder (`d:\Inundation`) and run these steps in order.

### Step 1: Create a virtual environment

```powershell
python -m venv .venv
```

*(Already done on this machine. Skip this step if the `.venv` folder exists.)*

### Step 2: Activate it

```powershell
.\.venv\Scripts\Activate.ps1
```

Your prompt should now start with `(.venv)`.

If PowerShell says *"running scripts is disabled on this system"*, run this once and then activate again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Step 3: Install the packages

```powershell
pip install -r requirements.txt
```

*(Already done on this machine.)*

### Step 4: Optional settings file

```powershell
copy .env.example .env
```

You can skip this step. The defaults work fully offline and send no data anywhere.
Create `.env` only if you want to change something, such as connecting a language model (see section 7).

### Step 5: Build the database from the PDFs

```powershell
python -m src.ingestion
```

This reads both PDFs, extracts every accident record, checks the result, and builds the search database.
It takes about a minute. The first time, it also downloads the search model.

A successful run ends like this:

```
Mines: 769 | incidents: 1006 (water-related: 73) | findings: 1537 | sources: 1063
  Coal Mines: 433 mines, 51 with water-related records
  Metal / Non-Coal Mines: 336 mines, 17 with water-related records
Vector index: ok | page passages for document chat: 1280 (semantic: ok)
Audit: ...\data\processed\extraction_audit.md
```

The "page passages" are every text page of both PDFs, split into searchable pieces for the chat.

*(Already done on this machine.)* Running it again does nothing unless the PDFs have changed.

- If it prints **"Vector index: unavailable"**, the model download failed. Everything still works with
  keyword search only. Run the same command again later to add semantic search.
- If it prints **"BLOCKED"**, read the message: it says exactly which PDF is missing or unreadable.
  The program never substitutes made-up data.

---

## 4. Running the application

Each time you want to use it:

```powershell
cd d:\Inundation
.\.venv\Scripts\Activate.ps1
streamlit run app.py --server.headless true
```

Then open **http://127.0.0.1:8501** in your browser.

- Use `127.0.0.1`, not `localhost`. The app listens on IPv4 only, and `localhost` may fail or be slow
  to connect.
- `--server.headless true` stops Streamlit asking for an email address on first launch. Without it,
  a non-interactive start can hang or fail.
- The app is reachable only from this computer, not from the network.

**To stop it:** press `Ctrl + C` in the PowerShell window, or close that window.

---

## 5. Using the application

### Home page
Choose **Coal Mines** or **Metal / Non-Coal Mines**. The counts shown are calculated from the extracted
records.

### Mine directory
- Type a mine name, district, state or owner in the search box. Small spelling mistakes are tolerated.
- Narrow the list with the **State** and **District** filters.
- By default only mines with water-related records are listed. Turn off the toggle to see every mine.
- Pick a mine from the drop-down or click a row in the table, then press **Open dashboard**.
- When two mines share a name, district, state and owner tell them apart.

### Mine dashboard (five tabs)

| Tab | What you see |
|---|---|
| **A. Overview** | Mine identity, location, number of recorded accidents, and a list of what information is missing. |
| **B. Inundation timeline** | A chart, then one card per accident with the official DGMS description, the findings and the exact PDF page. Expand **Source** to see the extracted text. |
| **C. Preventive findings** | Each precaution DGMS found was missing, which accident it relates to, the regulations cited, and the source page. |
| **D. Emergency response plan** | Press **Generate plan**. You get six stages of steps; each shows its basis, evidence, responsible role, priority, what to check first and which approvals are needed. Download it as Markdown or JSON. |
| **Search this mine's records** | Search inside the selected mine's records only. |

**Colour labels:**
- **Blue:** documented in the source PDF.
- **Amber:** a recommendation or inference generated by the system.
- **Grey:** not documented in the source.

### Inundation chat
Open **Inundation chat** from the sidebar, or click the round **Inundation chat** button in the bottom-right
corner of any other page. The chat page looks like a modern AI chat app. The left sidebar shows only the
chat tools: **Back to main menu** (arrow), **New chat**, **Recents** and **Memory**; the app's other pages
come back when you return to the main menu. The conversation fills the rest of the window. A new chat
starts with **"Where should we begin?"**, a large input box and four example questions you can click.

Choose the answer mode in the dropdown next to the message box (beside the send arrow):

**Research report** (default; about 30-60 seconds) - for questions such as *"I am a 50 year old, placed as
CMD at ECL, what are the inundation conditions there?"*. The assistant:

1. **Understands the question first:** role (CMD = Chairman-cum-Managing Director), organisation (ECL =
   Eastern Coalfields Limited), places, subject and intent. Personal details such as age are noted as
   context only. Anything it assumed is listed separately, and if nothing can be identified it asks you
   a clarifying question instead of guessing.
2. **Researches online:** several targeted searches, then it reads the most relevant pages (including PDF
   reports), keeping each source's link and publication date.
3. **Fact-checks against the two SANKET PDFs:** the key web findings are listed in a table and each is
   marked *supported*, *contradicted*, *qualified* or *not addressed* by the PDFs, with page references. If
   your question names an organisation, every water-related accident the PDFs record for that owner's mines
   is included.
4. **Writes the assessment:** executive summary, context and scope, findings, PDF fact-check,
   historical and current assessment, risk analysis, conclusions, follow-up actions, and sources and
   limitations. Web facts are cited **[W1]** and PDF facts **[S1]**.
5. **Shows charts and tables built only from PDF data:** for example, the organisation's water-related
   fatal accidents by year, and the list of those accidents with page numbers.

Under each report: the fact-check table, **Sources** (web links with dates, PDF pages), **Figures to
double-check** (numbers not found in the source they cite), and **Research steps** (how the question was
understood, assumptions, and the searches run). Claims about *current* conditions need recent dated
sources; otherwise the report says that current conditions could not be verified. The PDFs end in 2022.

Web results from the free search service vary from run to run. If it returns little, the report relies
more on the PDFs and says so.

**Quick answer** (about 5-15 seconds) - answers from the **internet only**: the question is searched on
the web, the top pages are read, and the AI model answers from those results with clickable **[W1]**
citations. The citations and numbers are checked instantly, as described below. The PDFs are not used.

**PDF answer** (about 5-10 seconds) - answers from the two SANKET PDFs first, with up to 20% web context.
Examples: "How many fatal accidents occurred in coal mines in 2019?" or "What happened at Sayal D
colliery?"

How PDF answers work:
- The answer is built from passages found in the PDFs. Each sentence taken from them ends with a marker
  such as **[S2]**. Open **Sources** under the answer to see the file, page and text for each marker.
- Up to **20%** of an answer may come from outside the two PDFs. While the PDFs are searched, the
  question is also searched on the internet (DuckDuckGo; only the question is sent, never PDF text).
  Web facts are cited as **[W1]**, **[W2]** with a clickable link under **Sources**, and are used for context
  the PDFs lack, such as how an accident happened or what an organisation is. General background knowledge
  is labelled *(general knowledge)*. Neither ever overrides the PDFs, and the 20% limit is enforced
  automatically: extra outside sentences are removed. Sentences the assistant infers rather than reads are
  labelled *(inference)*. If the PDFs don't answer at all, web facts are shown under "From internet
  sources (not from the SANKET documents)".
- **How an answer is checked:** the answer appears word by word as it is written. When it finishes, an
  instant check (no extra waiting) confirms that every citation points to a source that was actually
  retrieved, and that every number and date appears in the cited source. Failing sentences are removed.
  The answer then shows one of these labels:

  | Label | Meaning |
  |---|---|
  | **Citations checked** | Default (fast) mode: citations and numbers were checked; the meaning of each sentence was not checked by a second model. |
  | **Sources verified** | Thorough mode only: every statement was also checked by an independent model call. |
  | **Partly verified - unsupported statements removed** | Thorough mode: some statements failed and were removed. |
  | **Not established from the documents** | The PDFs do not contain the answer; nothing is guessed. |
  | **NOT verified - verification unavailable** | Thorough mode: the checking model could not be reached. |
  | **Passages only - no model available** | No language model; the best-matching PDF passages are shown instead. |

  The slower thorough mode (`SANKET_CHAT_VERIFY_MODE=thorough` in `.env`) adds a second model check and
  can redraft an answer; it typically adds 5-30 seconds per answer. Open **How this answer was checked** to
  see which sentences were removed, and why.
- For questions about inundation, drowning or water, the chat also receives the complete list of
  water-related records extracted from both PDFs (each row with its page), so counts and lists cover
  every record, not just the best-matching pages.
- **Follow-up questions** ("which of those were in Jharkhand?", "explain that further") are understood from
  the conversation.
- **Stop** interrupts an answer that is being written. No unverified answer is saved; the question shows
  as interrupted with a **Retry** button.
- In the sidebar: **New chat**, **Recents** (open, rename, delete, or clear all), and **Memory**.
- **Memory:** short notes of the topics you asked about, built from your own questions only. They are
  used only when you refer back to an earlier conversation ("earlier I asked about ..."), and never count as
  evidence. You can view them, delete them, or switch memory off.
- If the PDFs don't contain the answer, the assistant says so.
- With no working language model, the chat shows the best-matching PDF passages instead of a written answer.

The chat needs a language model (section 7) and sends the retrieved passages for each question to that
provider. Conversations are stored on this computer in `data/chat/`. Your conversations are tied to a
private link: the address bar shows `?u=...` after the first visit. Keep that link (for example as a
bookmark) to return to your conversations; anyone with the link can see them.

### Data audit page (sidebar)
Shows what was extracted from each PDF, which records are uncertain, and which mine names look like
possible duplicates that were deliberately *not* merged.

### Examples to try

| Search | Category | What it shows |
|---|---|---|
| `Sayal` | Coal | 2017 water inrush after a brick wall holding water failed |
| `Kuardih` | Coal | 2017 drowning after surface water entered the workings |
| `Chasnalla` | Coal | Two separate entries: the 1975 disaster (name only) and a located Dhanbad mine |
| `Turamdih` | Metal / Non-Coal | 2016 inrush of sludge and water at a uranium mine |

---

## 6. Rebuilding after the PDFs change

If the PDFs are replaced or updated:

```powershell
python -m src.ingestion --rebuild
```

Then restart the application.

---

## 7. Optional: connecting a language model

A language model is required for written answers in the document chat. It is optional for emergency
plans, which otherwise come from a built-in rule engine that quotes the records and sends nothing
anywhere.

**Current configuration on this machine:** `.env` uses **Groq** (`openai/gpt-oss-120b`, key in
`GROQ_API_KEY`) and falls back automatically to **OpenRouter** (`nvidia/nemotron-3-super-120b-a12b:free`,
key in `OPENROUTER_API_KEY`) if Groq fails before answering (rate limit, request too large, outage, bad key).
The OpenAI key in `.env` is not used.

```
SANKET_LLM_PROVIDER=groq
SANKET_LLM_MODEL=openai/gpt-oss-120b
SANKET_LLM_FALLBACK_PROVIDERS=openrouter
SANKET_OPENROUTER_MODEL=nvidia/nemotron-3-super-120b-a12b:free
SANKET_CHAT_REASONING_EFFORT=low
SANKET_ALLOW_EXTERNAL_LLM=true
SANKET_CHAT_WEB=true            # up to 20% of each chat answer from web search (DuckDuckGo, no key)
SANKET_CHAT_VERIFY_MODE=fast    # instant citation/number checks; "thorough" adds a slower model check
```

Both keys are free-tier keys. Free tiers have per-minute limits: when Groq's limit is reached, the
OpenRouter fallback answers. OpenRouter is slower (often 30 seconds to 2 minutes per answer instead of
5 to 30 seconds). Paid tiers remove most of these delays. Under each answer you can see which provider
answered.

To use a different provider, edit `.env`:

| Setting | Effect |
|---|---|
| `SANKET_LLM_PROVIDER=groq` | Uses Groq Cloud (`GROQ_API_KEY`). Data goes to an external service. |
| `SANKET_LLM_PROVIDER=openrouter` | Uses OpenRouter (`OPENROUTER_API_KEY`). Data goes to an external service. |
| `SANKET_LLM_PROVIDER=ollama` | Uses a model on your own machine through Ollama. Data stays local. |
| `SANKET_LLM_PROVIDER=anthropic` | Uses Claude through the Anthropic API. Data goes to an external service. |
| `SANKET_LLM_PROVIDER=openai` | Uses OpenAI or a compatible server. |
| `SANKET_LLM_FALLBACK_PROVIDERS=...` | Comma-separated providers to try, in order, if the first one fails. |

For an external provider you must also set **both** of these:

```
GROQ_API_KEY=your-key            # or OPENROUTER_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY
SANKET_ALLOW_EXTERNAL_LLM=true
```

Until `SANKET_ALLOW_EXTERNAL_LLM=true` is set, external providers are blocked. The sidebar always states
where data is going. Victims' names are removed from the evidence sent for emergency plans.

**Tested:** the chat has been run end to end against Groq, the OpenRouter fallback, and the HTTP API
(section 8). Emergency plans written by a language model (rather than the rule engine) have not been
reviewed against a live model.

---

## 8. Optional: the chat HTTP API

The same verified chat is available as a web API with streaming answers (Server-Sent Events), for use by
other applications. It is off unless you set a secret:

```powershell
# in .env:   SANKET_API_TOKEN=choose-a-long-random-secret
uvicorn src.api:app --host 127.0.0.1 --port 8600
```

Every request needs two headers: `Authorization: Bearer <SANKET_API_TOKEN>` and
`X-Sanket-User: <a private user token, 16+ characters>`. Each user token sees only its own conversations.

| Method and path | Purpose |
|---|---|
| `POST /api/chat/stream` | Ask a question. Body: `{"message": "...", "conversation_id": "<optional>", "category": "coal" or "metal_non_coal" (optional)}`. Streams events. |
| `GET /api/conversations` | List your conversations. |
| `POST /api/conversations` | Create one (optional `{"title": "..."}`). |
| `GET /api/conversations/{id}/messages` | Messages with status, citations and check details. |
| `PATCH /api/conversations/{id}` | Rename: `{"title": "..."}`. |
| `DELETE /api/conversations/{id}` / `DELETE /api/conversations` | Delete one / delete all. |
| `GET` / `PUT` / `DELETE /api/memory` | View memory notes, switch memory on/off (`{"enabled": false}`), delete notes. |
| `GET /api/health` | Status (no headers needed). |

Streamed events: `status` (progress), `token` (draft text, **not verified**), `citation` (a source used by
the final answer), `done` (the final, checked answer and its label), `error`. Only `done` contains the
answer to display as final.

### Optional: Redis cache

Repeated questions are answered from a cache, but only for the same document version: changing either
PDF and re-running ingestion invalidates every cached answer. The cache is a local file by default.
To use Redis instead, set `SANKET_REDIS_URL=redis://localhost:6379/0`; if Redis cannot be reached, the
local file cache is used.

---

## 9. Checking that everything works

```powershell
python -m pytest tests -q        # 116 automated tests, about 30 seconds (no API calls are made)
python -m src.evaluation         # accuracy checks on the real data
```

---

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `python` is not recognised | Install Python 3.11+ from python.org and tick "Add Python to PATH". |
| "running scripts is disabled" on activate | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then activate again. |
| App says "dataset has not been built" | Run `python -m src.ingestion` and reload the page. |
| Browser cannot reach `localhost:8501` | Use `http://127.0.0.1:8501`. |
| Startup hangs at "Email:" | Restart with `--server.headless true`, or just press Enter. |
| Port 8501 already in use | Add `--server.port 8502` and open `http://127.0.0.1:8502`. |
| "Vector index: unavailable" | Search model download failed. The app still works; rerun `python -m src.ingestion` later. |
| A plan says "Language model not used" | The model was unavailable or blocked, so the offline plan is shown. Check `.env`. |
| Chat says "no API credits left" | Add credits to that provider's account (billing page), then ask again. No restart needed. |
| Chat shows passages instead of an answer | No usable model. Read the reason in the first line of the reply. |
| Chat answers are slow (1-2 minutes) | Groq's free-tier limit was reached and the slower OpenRouter fallback answered. Wait a minute between questions, or use a paid tier. |
| "Too many questions in the last minute" | The chat's own limit (`SANKET_CHAT_RATE_LIMIT_PER_MIN`, default 12). Wait a moment. |
| Answer labelled "NOT verified" | The checking model could not be reached. Ask again later, or check the cited pages yourself. |
| No web sources in answers | Web search is off (`SANKET_CHAT_WEB=false`), there is no internet connection, or the results added nothing the PDFs lacked. The reason, if any, is shown under the answer. |
| A question shows "Interrupted" | It was stopped or the page was closed while answering. Press **Retry**. |
| Previous conversations are missing | Open the app with the same `?u=...` link you used before. |
| Chat button not visible | Reload the page. It sits in the bottom-right corner, above the page content. |

---

## 11. Quick reference

```powershell
# one-time
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.ingestion

# every time
.\.venv\Scripts\Activate.ps1
streamlit run app.py --server.headless true
# open http://127.0.0.1:8501, stop with Ctrl + C
# chat: the "Inundation chat" button, bottom-right of every page

# optional chat API (needs SANKET_API_TOKEN in .env)
uvicorn src.api:app --host 127.0.0.1 --port 8600
```
