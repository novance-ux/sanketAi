"""Shared Streamlit helpers: page chrome, cached resources, labels and small components.

No extraction or indexing happens here - the UI only reads the database that
`python -m src.ingestion` has already built.
"""
from __future__ import annotations

import html
import json
import re

import streamlit as st

from .agent import SanketAgent
from .config import get_settings
from .database import Database
from .llm import data_flow_notice
from .retrievers import Retriever
from .schemas import CATEGORY_LABELS, Mine

NOT_DOCUMENTED = "Not documented in source"
DIRECTORY_PAGE = "pages/1_Mine_Directory.py"
DASHBOARD_PAGE = "pages/2_Mine_Intelligence.py"
AUDIT_PAGE = "pages/3_Data_Audit.py"

# Validated categorical palette (colour-blind safe on a light surface); each series also
# gets its own marker symbol so identity never depends on colour alone.
SERIES_STYLE = {
    "irruption": {"label": "Irruption / inundation", "color": "#2a78d6", "symbol": "circle"},
    "drowning": {"label": "Drowning in water", "color": "#eb6834", "symbol": "diamond"},
    "water_keyword": {"label": "Water in narrative", "color": "#1baf7a", "symbol": "square"},
    "none": {"label": "Other fatal accident", "color": "#8d8c86", "symbol": "x"},
}

_CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1180px;}
h1, h2, h3 {letter-spacing: -0.01em;}
.sk-sub {color: #52514e; font-size: 1.05rem; margin-top: -0.6rem;}
.sk-card {border: 1px solid #d9d8d2; border-radius: 6px; padding: 1.1rem 1.2rem; background: #ffffff;}
.sk-card h3 {margin: 0 0 .3rem 0; font-size: 1.2rem;}
.sk-big {font-size: 2rem; font-weight: 650; line-height: 1.1; color: #0b0b0b;}
.sk-muted {color: #6b6a65; font-size: .88rem;}
.sk-badge {display: inline-block; padding: 1px 8px; border-radius: 3px; font-size: .74rem; font-weight: 600;
           letter-spacing: .02em; margin-right: 6px; border: 1px solid; white-space: nowrap;}
.sk-doc {color: #184f95; border-color: #86b6ef; background: #eef5fd;}
.sk-ai {color: #7a4b00; border-color: #e0b25a; background: #fdf5e3;}
.sk-missing {color: #52514e; border-color: #c3c2b7; background: #f4f4f1;}
.sk-warn {color: #8a2a12; border-color: #ec835a; background: #fdf0ea;}
.sk-quote {border-left: 3px solid #86b6ef; padding: .35rem .9rem; background: #f7fafe; margin: .35rem 0 .6rem 0;
           font-size: .95rem; white-space: pre-wrap;}
.sk-banner {border: 1px solid #c3c2b7; border-left: 4px solid #52514e; padding: .6rem .9rem; background: #f8f8f5;
            font-size: .9rem; border-radius: 3px; margin-bottom: .8rem;}
</style>
"""


def page_setup(title: str) -> None:
    st.set_page_config(page_title=f"SANKET AI - {title}", layout="wide", initial_sidebar_state="expanded")
    st.markdown(_CSS, unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("**SANKET AI**")
        st.caption("Historical mine inundation intelligence and emergency decision support")
        st.page_link("app.py", label="Home")
        st.page_link(DIRECTORY_PAGE, label="Mine directory")
        st.page_link(DASHBOARD_PAGE, label="Mine intelligence dashboard")
        st.page_link(AUDIT_PAGE, label="Data audit and system status")
        st.divider()
        st.caption("Data flow")
        st.caption(data_flow_notice())
        st.divider()
        st.caption("Decision-support aid. Not an official order; verify every item with qualified personnel.")


# --------------------------------------------------------------------------- cached access
@st.cache_resource(show_spinner=False)
def get_retriever() -> Retriever:
    return Retriever.from_settings(get_settings())


@st.cache_resource(show_spinner=False)
def get_agent() -> SanketAgent:
    return SanketAgent(get_retriever(), get_settings())


def get_db() -> Database:
    return get_retriever().db


def db_version() -> float:
    """Cache key that changes whenever ingestion rewrites the database."""
    path = get_settings().db_path
    return path.stat().st_mtime if path.is_file() else 0.0


def require_database() -> Database:
    """Stop the page with recovery instructions when the dataset has not been built."""
    db = get_db()
    if not db.exists():
        st.error("The SANKET dataset has not been built yet.")
        st.markdown(
            "1. Place the two SANKET PDF volumes (coal and non-coal) in the project folder or in `data/raw/`.\n"
            "2. Run `python -m src.ingestion` from the project folder.\n"
            "3. Reload this page.\n\n"
            "No sample data is shown in place of the real records."
        )
        st.stop()
    return db


@st.cache_data(show_spinner=False)
def load_category_counts(_version: float) -> dict:
    return get_db().category_counts()


@st.cache_data(show_spinner=False)
def load_mines(_version: float, category: str | None, water_only: bool) -> list[dict]:
    return [m.model_dump() for m in get_db().list_mines(category, water_only=water_only)]


@st.cache_data(show_spinner=False)
def load_locations(_version: float, category: str | None) -> dict[str, list[str]]:
    return get_db().locations(category)


@st.cache_data(show_spinner=False)
def load_report(_version: float) -> dict | None:
    path = get_settings().report_json
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


# --------------------------------------------------------------------------- formatting
def md_escape(text: str | None) -> str:
    """Show document text literally inside st.markdown (no markdown/LaTeX interpretation)."""
    return re.sub(r"([\\`*_{}\[\]<>#$|~])", r"\\\1", text or "")


def badge(kind: str, text: str) -> str:
    return f'<span class="sk-badge sk-{kind}">{html.escape(text)}</span>'


def quote_block(text: str | None) -> None:
    st.markdown(f'<div class="sk-quote">{html.escape(text or "")}</div>', unsafe_allow_html=True)


def banner(text: str) -> None:
    st.markdown(f'<div class="sk-banner">{html.escape(text)}</div>', unsafe_allow_html=True)


def place_of(mine: Mine | dict) -> str:
    d = mine if isinstance(mine, dict) else mine.model_dump()
    return ", ".join(x for x in (d.get("district"), d.get("state")) if x) or "location not documented"


def mine_label(mine: Mine | dict) -> str:
    """Unambiguous selector label: name, location, owner and record count."""
    d = mine if isinstance(mine, dict) else mine.model_dump()
    owner = f" | {d['operator']}" if d.get("operator") else ""
    return (f"{d['mine_name']} - {place_of(d)}{owner} "
            f"[{d['water_incident_count']} water-related / {d['incident_count']} records]")


def category_label(category: str | None) -> str:
    return CATEGORY_LABELS.get(category or "", "All categories")


def open_mine(mine_id: str) -> None:
    st.session_state["mine_id"] = mine_id
    st.switch_page(DASHBOARD_PAGE)
