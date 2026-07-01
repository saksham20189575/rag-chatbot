"""Streamlit chat UI — Phase 5 (Option A).

Run the API first::

    uvicorn src.api.main:app --reload

Then start the UI::

    streamlit run src/ui/app.py

Optional: set ``CHAT_API_URL`` (default ``http://localhost:8000``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Streamlit executes this file directly; ensure project root is on sys.path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from dotenv import load_dotenv

from src.constants import DISCLAIMER, SCHEME_NAMES
from src.ui.api_client import ChatApiError, get_health, post_chat, resolve_api_base_url

load_dotenv(PROJECT_ROOT / ".env")

APP_TITLE = "Mutual Fund FAQ Assistant — HDFC Schemes"

EXAMPLE_QUESTIONS: tuple[str, ...] = (
    "What is the expense ratio of HDFC Large Cap Fund Direct Growth?",
    "What is the exit load for HDFC Gold ETF Fund of Fund?",
    "What is the minimum SIP for HDFC Small Cap Fund Direct Growth?",
)

GROWW_GREEN = "#00D09C"


def _inject_styles() -> None:
    """Theme-aware accents — uses Streamlit CSS variables for light/dark compatibility."""
    st.markdown(
        f"""
        <style>
        div[data-testid="stHorizontalBlock"] button {{
            border-color: {GROWW_GREEN} !important;
        }}
        div[data-testid="stHorizontalBlock"] button:hover {{
            border-color: {GROWW_GREEN} !important;
            color: {GROWW_GREEN} !important;
        }}
        a[href*="groww.in"] {{
            color: {GROWW_GREEN} !important;
            font-weight: 600;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None
    if "welcome_shown" not in st.session_state:
        st.session_state.welcome_shown = False


def _render_welcome() -> None:
    with st.container(border=True):
        st.markdown(
            "**Welcome!** Ask factual questions about these five HDFC schemes on Groww:"
        )
        for name in SCHEME_NAMES.values():
            st.markdown(f"- {name}")
        st.markdown(
            "Try an example below or type your own question. "
            "Please do not share personal information (PAN, phone, folio, etc.)."
        )


def _render_assistant_message(payload: dict[str, Any]) -> None:
    is_refusal = payload.get("type") == "refusal"
    text = payload.get("text", "")
    citation_url = payload.get("citation_url", "")
    last_updated = payload.get("last_updated", "")

    if is_refusal:
        st.warning(text)
    else:
        st.markdown(text)

    if citation_url:
        st.markdown(f"[View on Groww]({citation_url})")

    if last_updated:
        st.caption(f"Last updated from sources: {last_updated}")


def _process_query(query: str) -> None:
    query = query.strip()
    if not query:
        return

    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Looking up fund facts…"):
        try:
            payload = post_chat(query)
        except ChatApiError as exc:
            payload = {
                "type": "refusal",
                "text": str(exc),
                "citation_url": "",
                "last_updated": "",
                "disclaimer": DISCLAIMER,
            }

    st.session_state.messages.append({"role": "assistant", "payload": payload})


def main() -> None:
    st.set_page_config(
        page_title="Mutual Fund FAQ Assistant",
        page_icon="💬",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    _init_session_state()
    _inject_styles()

    st.title(APP_TITLE)
    st.info(DISCLAIMER)

    api_url = resolve_api_base_url()
    try:
        health = get_health(api_url)
        chunk_count = health.get("index", {}).get("chunk_count", "?")
        st.caption(f"Connected to API · {chunk_count} indexed chunks")
    except ChatApiError as exc:
        st.error(
            f"Cannot connect to the chat API at `{api_url}`. "
            f"Start it with: `uvicorn src.api.main:app --reload`\n\n{exc}"
        )

    if not st.session_state.welcome_shown:
        _render_welcome()
        st.session_state.welcome_shown = True

    st.markdown("**Example questions**")
    chip_cols = st.columns(len(EXAMPLE_QUESTIONS))
    for idx, question in enumerate(EXAMPLE_QUESTIONS):
        label = question if len(question) <= 42 else question[:39] + "…"
        if chip_cols[idx].button(label, key=f"example_{idx}", use_container_width=True):
            st.session_state.pending_query = question
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                _render_assistant_message(message["payload"])

    pending = st.session_state.pending_query
    if pending:
        st.session_state.pending_query = None
        _process_query(pending)
        st.rerun()

    if prompt := st.chat_input("Ask a factual question about an HDFC scheme…"):
        _process_query(prompt)
        st.rerun()


if __name__ == "__main__":
    main()
