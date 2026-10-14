# SANKET AI: How to Run It

This guide covers what SANKET AI does, what to set up before the first run, how to start it, and how to use it.
For architecture, data schema and test details, see [README.md](README.md).

---

## 1. What it does

SANKET AI is a local web application for mining officials (District Magistrates, mine managers, safety
officers). It is meant for officers taking charge of mines in a region they do not know.

You choose a mine, and the application shows:

1. **Its recorded history of water-related accidents.** These are inundations, irruptions of water and
   drownings, taken from the two DGMS SANKET PDF volumes (coal and non-coal mines).
2. **What the DGMS inquiry found.** For each accident, these are the precautions that DGMS said would have
   prevented it ("Had ... been done, this accident could have been averted"). They describe what was
   *missing at the time*. They are not proof that anything was installed afterwards.
3. **A step-by-step emergency response plan** in six stages, from detecting the problem through to recovery.
   Each step says whether it is based on that mine's own records or on general safety practice.

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
| Internet | Needed once to install packages and download a 90 MB search model. After that it runs offline. |
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
Vector index: ok
Audit: ...\data\processed\extraction_audit.md
```

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

By default the plan is assembled by a built-in rule engine, which quotes the records directly and sends
nothing anywhere. To have a language model write the plan instead, edit `.env`:

| Setting | Effect |
|---|---|
| `SANKET_LLM_PROVIDER=ollama` | Uses a model on your own machine through Ollama. Data stays local. |
| `SANKET_LLM_PROVIDER=anthropic` | Uses Claude through the Anthropic API. Data goes to an external service. |
| `SANKET_LLM_PROVIDER=openai` | Uses OpenAI or a compatible server. |

For an external provider you must also set **both** of these:

```
ANTHROPIC_API_KEY=your-key            # or OPENAI_API_KEY
SANKET_ALLOW_EXTERNAL_LLM=true
```

Until `SANKET_ALLOW_EXTERNAL_LLM=true` is set, external providers are blocked. The sidebar always states
where data is going. Victims' names are removed before anything is sent.

**Untested path:** the language-model path has not been run against a live model. The offline engine is
the tested default.

---

## 8. Checking that everything works

```powershell
python -m pytest tests -q        # 65 automated tests, about 30 seconds
python -m src.evaluation         # accuracy checks on the real data
```

---

## 9. Troubleshooting

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

---

## 10. Quick reference

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
```
