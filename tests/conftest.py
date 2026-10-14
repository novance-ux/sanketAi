"""Shared fixtures.

Two kinds of data are used and never mixed:

* ``real_*`` fixtures read the database built from the actual SANKET PDFs. Tests that need
  them are skipped when the PDFs are not present.
* ``synthetic_*`` fixtures build throw-away records in a temp directory to exercise logic.
  Their mine names are obviously artificial and they never touch ``data/processed``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent import SanketAgent  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.data_normalizer import normalize  # noqa: E402
from src.database import Database  # noqa: E402
from src.ingestion import build_chunks, discover_pdfs, run_ingestion  # noqa: E402
from src.pdf_extractor import (  # noqa: E402
    ExtractionResult,
    PdfDocument,
    RawCourtRow,
    RawFatalRecord,
    RawMajorRow,
    extract_document,
)
from src.retrievers import Retriever  # noqa: E402
from src.vector_store import VectorStore  # noqa: E402


@pytest.fixture(autouse=True)
def _offline_llm(monkeypatch):
    """Tests never call a real model or send data anywhere."""
    monkeypatch.setenv("SANKET_LLM_PROVIDER", "offline")
    monkeypatch.setenv("SANKET_ALLOW_EXTERNAL_LLM", "false")


# --------------------------------------------------------------------------- real data
@pytest.fixture(scope="session")
def real_pdfs() -> list[Path]:
    pdfs = discover_pdfs(get_settings())
    if len(pdfs) < 2:
        pytest.skip("The two SANKET PDFs are not available in the configured folders.")
    return pdfs


@pytest.fixture(scope="session")
def real_extractions(real_pdfs) -> dict[str, ExtractionResult]:
    results = [extract_document(p) for p in real_pdfs]
    return {r.document.category: r for r in results}


@pytest.fixture(scope="session")
def real_settings(real_pdfs):
    settings = get_settings()
    if not Database(settings.db_path).exists():
        run_ingestion(settings, skip_vectors=True)
    return settings


@pytest.fixture(scope="session")
def real_db(real_settings) -> Database:
    return Database(real_settings.db_path)


@pytest.fixture(scope="session")
def real_retriever(real_settings) -> Retriever:
    return Retriever(Database(real_settings.db_path),
                     VectorStore(real_settings.vector_dir, real_settings.embedding_backend), real_settings)


@pytest.fixture()
def real_agent(real_retriever) -> SanketAgent:
    return SanketAgent(real_retriever, get_settings())


def find_mine(db: Database, name: str, category: str, *, located: bool | None = None):
    for m in db.list_mines(category):
        if m.mine_name.upper() == name.upper() and (located is None or (m.state is not None) == located):
            return m
    raise AssertionError(f"{name} not found")


# --------------------------------------------------------------------------- synthetic data
def _doc(category: str, filename: str) -> PdfDocument:
    return PdfDocument(Path(filename), filename, "0" * 64, 10, category, "synthetic", [])


def fatal(serial, year, date_text, mine, state, district, code, label, narrative, findings="", owner="TEST OWNER LTD",
          category="coal", filename="synthetic_coal.pdf") -> RawFatalRecord:
    return RawFatalRecord(
        filename=filename, category=category, year=year, serial=serial, date_text=date_text, mine_name=mine,
        owner=owner, district=district, state=state, persons_killed=["1. A. Person, Pump Operator, Male, 40 Years"],
        narrative=narrative, findings_text=findings, cause_code=code, cause_label=label, parent_code="0900",
        parent_label="Other Causes", pages=[5], source_text=f"{serial}. Date - {date_text} Mine - {mine}\n{narrative}\n{findings}",
    )


@pytest.fixture()
def synthetic_results() -> list[ExtractionResult]:
    coal = ExtractionResult(document=_doc("coal", "synthetic_coal.pdf"))
    coal.fatal_records = [
        fatal(1, 2017, "16.08.17", "TESTMINE ALPHA", "Teststate", "Testdistrict", "0991", "Irruption of Water",
              "While a brick wall holding water failed, inrush of water of about 4.5m head drowned two persons.",
              "Had,\n\ni) the dam been of standard design as required under Regulation 129(1) of the Coal Mines "
              "Regulations, 1957,\n\nii) persons not been deployed below water logged workings,\n\n"
              "this accident could have been averted."),
        fatal(2, 2018, "02.03.18", "TESTMINE ALPHA", "Teststate", "Testdistrict", "0335", "Dumpers",
              "While a dumper was reversing it ran over a person.", "Had, the dumper been reversed carefully, "
              "this accident could have been averted."),
        fatal(3, 2018, "05.05.18", "TESTMINE ALPHA", "Otherstate", "Otherdistrict", "0993", "Drowning in Water",
              "While a person was near a sump he fell and drowned.", ""),
        fatal(4, 2019, "07.07.19", "TESTMINE BETA OC-II", "Teststate", "Testdistrict", "0112", "Fall of Sides",
              "While a person was working a side fell.", ""),
        fatal(5, 2019, "08.07.19", "TESTMINE BETA OC-III", "Teststate", "Testdistrict", "0993", "Drowning in Water",
              "While a person was at the pond he drowned.", ""),
    ]
    coal.major_rows = [
        RawMajorRow("synthetic_coal.pdf", "coal", 1, "27-12-1975", "Testmine Gamma", 375, 0, "Irruption of Water", 8,
                    "1 | 27-12-1975 | Testmine Gamma | 375 | 0 | Irruption of Water"),
        RawMajorRow("synthetic_coal.pdf", "coal", 2, "16-09-1976", "Testmine Delta", 10, 0, "Irruption of Water", 8,
                    "2 | 16-09-1976 | Testmine Delta | 10 | 0 | Irruption of Water"),
        RawMajorRow("synthetic_coal.pdf", "coal", 3, "15-06-2005", "Testmine Dellta", 14, 0, "Irruption of Water", 8,
                    "3 | 15-06-2005 | Testmine Dellta | 14 | 0 | Irruption of Water"),
        RawMajorRow("synthetic_coal.pdf", "coal", 4, "01-01-1980", "Testmine Alpha", 5, 0, "Fall of Roof", 8,
                    "4 | 01-01-1980 | Testmine Alpha | 5 | 0 | Fall of Roof"),
    ]
    coal.court_rows = [
        RawCourtRow("synthetic_coal.pdf", "coal", 1, None, "27/12/75", "Testmine Gamma", 375, "Inundation",
                    "Justice X", 9, " 1  27/12/75  Testmine Gamma   375   Justice X\n   (Inundation)"),
    ]
    non_coal = ExtractionResult(document=_doc("metal_non_coal", "synthetic_noncoal.pdf"))
    non_coal.fatal_records = [
        fatal(1, 2016, "28.05.16", "TESTMINE ALPHA", "Teststate", "Testdistrict", "0991", "Irruption of Water",
              "While cleaning, sludge with water flowed into the decline.", "", category="metal_non_coal",
              filename="synthetic_noncoal.pdf"),
    ]
    non_coal.major_rows = [
        RawMajorRow("synthetic_noncoal.pdf", "metal_non_coal", 1, "05-11-14", "Testmine Mica", 5, 0,
                    "Irruption of Water", 3, "1 | 05-11-14 | Testmine Mica | 5 | 0 | Irruption of Water"),
    ]
    return [coal, non_coal]


@pytest.fixture()
def synthetic_dataset(synthetic_results):
    return normalize(synthetic_results, {})


@pytest.fixture()
def synthetic_retriever(synthetic_dataset, tmp_path) -> Retriever:
    db = Database(tmp_path / "synthetic.sqlite3")
    ds = synthetic_dataset
    db.write(mines=ds.mines, incidents=ds.incidents, findings=ds.findings, sources=ds.sources,
             chunks=build_chunks(ds), documents=[], meta={"fingerprint": "synthetic"})
    return Retriever(db, None, get_settings())
