"""Inundation chat - the research assistant (full-page conversational workspace)."""
from __future__ import annotations

from src.chat_ui import render_bot_page
from src.ui import page_setup, require_database

page_setup("Inundation chat", floating_chat=False, wide_page=True, show_nav=False)
require_database()
render_bot_page()
