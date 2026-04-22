from __future__ import annotations

import copy
import html
import json
import time
from datetime import datetime
from typing import Any

import streamlit as st

from src.pipeline.rag_pipeline import RAGPipeline

st.set_page_config(
    page_title="AcaIntel AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _inject_styles(*, dark: bool) -> None:
    if dark:
        bg = "#0b1020"
        surface = "#121a2b"
        surface_2 = "#182338"
        text = "#f3f6fb"
        muted = "#9aa7bd"
        border = "#26324a"
        accent = "#7c3aed"
        accent_2 = "#06b6d4"
        bubble_user = "#1d4ed8"
        bubble_assistant = "#111827"
        success = "#10b981"
        badge_budget_bg = "rgba(6,182,212,0.14)"
        badge_budget_text = "#67e8f9"
        badge_election_bg = "rgba(124,58,237,0.18)"
        badge_election_text = "#c4b5fd"
        badge_mixed_bg = "rgba(16,185,129,0.16)"
        badge_mixed_text = "#86efac"
        chat_in_bg = "#0f172a"
        chat_in_border = "#38bdf8"
        chat_in_border_soft = "#334155"
        chat_in_placeholder = "#94a3b8"
        chat_in_text = "#f8fafc"
        chat_in_caret = "#38bdf8"
    else:
        bg = "#f5f7fb"
        surface = "#ffffff"
        surface_2 = "#eef2ff"
        text = "#142033"
        muted = "#5f6b7a"
        border = "#dbe3f0"
        accent = "#6d28d9"
        accent_2 = "#0891b2"
        bubble_user = "#dbeafe"
        bubble_assistant = "#ffffff"
        success = "#059669"
        badge_budget_bg = "#e0f7ff"
        badge_budget_text = "#0f766e"
        badge_election_bg = "#efe7ff"
        badge_election_text = "#6d28d9"
        badge_mixed_bg = "#dcfce7"
        badge_mixed_text = "#166534"
        chat_in_bg = "#ffffff"
        chat_in_border = "#6d28d9"
        chat_in_border_soft = "#cbd5e1"
        chat_in_placeholder = "#475569"
        chat_in_text = "#142033"
        chat_in_caret = "#6d28d9"

    st.markdown(
        f"""
        <style>
            .stApp {{
                background:
                    radial-gradient(circle at top left, rgba(124,58,237,0.12), transparent 22%),
                    radial-gradient(circle at top right, rgba(6,182,212,0.10), transparent 18%),
                    {bg};
                color: {text};
            }}

            /* Keep Streamlit’s base theme from fighting the light/dark toggle (real-time rerun). */
            section[data-testid="stMain"], section[data-testid="stMain"] p, section[data-testid="stMain"] span,
            section[data-testid="stMain"] label, section[data-testid="stMain"] li {{
                color: {text} !important;
            }}

            [data-testid="stCaption"] {{
                color: {muted} !important;
            }}

            [data-testid="stMetricLabel"] {{
                color: {muted} !important;
            }}
            [data-testid="stMetricValue"] {{
                color: {text} !important;
            }}

            [data-testid="stHeader"] {{
                background: transparent;
            }}

            section[data-testid="stSidebar"] {{
                background: linear-gradient(180deg, {surface}, {surface_2});
                border-right: 1px solid {border};
                color: {text} !important;
            }}
            section[data-testid="stSidebar"] p,
            section[data-testid="stSidebar"] span,
            section[data-testid="stSidebar"] label,
            section[data-testid="stSidebar"] li,
            section[data-testid="stSidebar"] small {{
                color: {text} !important;
            }}
            section[data-testid="stSidebar"] [data-testid="stCaption"] {{
                color: {muted} !important;
            }}
            /* Don’t let sidebar text rules steal primary / download button contrast */
            section[data-testid="stSidebar"] .stButton button[kind="primary"],
            section[data-testid="stSidebar"] button[data-testid="baseButton-primary"] {{
                color: #ffffff !important;
            }}
            section[data-testid="stSidebar"] .stDownloadButton button {{
                color: {text} !important;
            }}

            .block-container {{
                padding-top: 1.2rem !important;
                padding-bottom: 4rem !important;
                max-width: 82rem !important;
            }}

            h1, h2, h3 {{
                color: {text} !important;
                letter-spacing: -0.02em;
            }}

            .hero-card {{
                background: linear-gradient(135deg, {surface}, {surface_2});
                border: 1px solid {border};
                border-radius: 22px;
                padding: 1.4rem 1.4rem 1.2rem 1.4rem;
                margin-bottom: 1rem;
                box-shadow: 0 10px 30px rgba(0,0,0,0.10);
            }}

            .hero-title {{
                font-size: 2rem;
                font-weight: 800;
                color: {text};
                margin-bottom: 0.2rem;
            }}

            .hero-sub {{
                color: {muted};
                font-size: 0.98rem;
                margin-bottom: 1rem;
            }}

            .pill-row {{
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
                margin-top: 0.4rem;
            }}

            .pill {{
                padding: 0.38rem 0.72rem;
                border-radius: 999px;
                border: 1px solid {border};
                background: {"rgba(255,255,255,0.08)" if dark else "rgba(15,23,42,0.04)"};
                color: {text};
                font-size: 0.82rem;
            }}

            .stat-card {{
                background: linear-gradient(180deg, {surface}, {surface_2});
                border: 1px solid {border};
                border-radius: 18px;
                padding: 0.9rem 1rem;
                height: 100%;
            }}

            .stat-label {{
                color: {muted};
                font-size: 0.8rem;
                margin-bottom: 0.2rem;
            }}

            .stat-value {{
                color: {text};
                font-size: 1.1rem;
                font-weight: 700;
            }}

            [data-testid="stChatMessage"] {{
                background: transparent !important;
                padding-top: 0.45rem;
                padding-bottom: 0.45rem;
            }}

            .assistant-bubble {{
                background: {bubble_assistant};
                border: 1px solid {border};
                border-radius: 18px;
                padding: 0.95rem 1rem;
                color: {text};
                box-shadow: 0 8px 22px rgba(0,0,0,0.08);
            }}

            .user-bubble {{
                background: {bubble_user};
                border: 1px solid transparent;
                border-radius: 18px;
                padding: 0.95rem 1rem;
                color: {"#ffffff" if dark else "#142033"};
                box-shadow: 0 8px 22px rgba(0,0,0,0.06);
            }}

            .section-label {{
                font-size: 0.82rem;
                color: {muted};
                margin-bottom: 0.35rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                font-weight: 700;
            }}

            .mini-note {{
                color: {muted};
                font-size: 0.82rem;
            }}

            .footer-note {{
                margin-top: 1.8rem;
                text-align: center;
                color: {muted};
                font-size: 0.82rem;
            }}

            .stButton button[kind="primary"] {{
                background: linear-gradient(90deg, {accent}, {accent_2}) !important;
                border: none !important;
                color: white !important;
                border-radius: 12px !important;
                font-weight: 700 !important;
            }}

            .stButton button {{
                border-radius: 12px !important;
            }}

            /* Chat composer: readable width + high contrast vs page background */
            [data-testid="stChatInput"] {{
                max-width: 48rem !important;
                margin-left: auto !important;
                margin-right: auto !important;
                margin-top: 0.5rem !important;
                background: {chat_in_bg} !important;
                border: 2px solid {chat_in_border} !important;
                border-radius: 16px !important;
                box-shadow:
                    0 0 0 1px {chat_in_border_soft},
                    0 10px 40px rgba(0, 0, 0, {"0.35" if dark else "0.12"});
            }}

            [data-testid="stChatInput"]:focus-within {{
                border-color: {accent_2} !important;
                box-shadow:
                    0 0 0 3px rgba(6, 182, 212, {"0.35" if dark else "0.25"}),
                    0 12px 36px rgba(0, 0, 0, {"0.4" if dark else "0.14"});
            }}

            /*
             * Chat input — typed text lives here. Streamlit’s theme sets textarea `color` to the
             * app surface tone (#121a2b in dark mode), which fought our fill color; keep both
             * `color` and `-webkit-text-fill-color` identical and use high specificity.
             */
            html body [data-testid="stChatInput"] textarea,
            html body [data-testid="stChatInput"] textarea:focus {{
                color: {chat_in_text} !important;
                -webkit-text-fill-color: {chat_in_text} !important;
                caret-color: {chat_in_caret} !important;
                background: transparent !important;
                font-size: 1.05rem !important;
                font-weight: 600 !important;
                line-height: 1.55 !important;
                min-height: 3.25rem !important;
                opacity: 1 !important;
            }}

            html body [data-testid="stChatInput"] textarea::placeholder {{
                color: {chat_in_placeholder} !important;
                -webkit-text-fill-color: {chat_in_placeholder} !important;
                opacity: 1 !important;
                font-weight: 500 !important;
            }}

            div[data-testid="stExpander"] {{
                border: 1px solid {border} !important;
                border-radius: 14px !important;
                overflow: hidden;
                background: {surface};
                color: {text} !important;
            }}
            div[data-testid="stExpander"] summary,
            div[data-testid="stExpander"] summary span {{
                color: {text} !important;
            }}

            .sidebar-title {{
                font-size: 1.15rem;
                font-weight: 800;
                color: {text} !important;
                margin-bottom: 0.15rem;
            }}

            .sidebar-sub {{
                color: {muted} !important;
                font-size: 0.84rem;
                margin-bottom: 0.9rem;
            }}

            .status-ok {{
                display: inline-block;
                background: rgba(16,185,129,0.12);
                color: {success};
                border: 1px solid rgba(16,185,129,0.28);
                padding: 0.35rem 0.65rem;
                border-radius: 999px;
                font-size: 0.78rem;
                font-weight: 700;
            }}

            .history-card {{
                border: 1px solid {border};
                background: {surface};
                border-radius: 14px;
                padding: 0.7rem 0.8rem;
                margin-bottom: 0.55rem;
            }}

            .history-title {{
                font-size: 0.82rem;
                font-weight: 700;
                color: {text};
                margin-bottom: 0.2rem;
            }}

            .history-meta {{
                color: {muted};
                font-size: 0.75rem;
            }}

            .source-badge-row {{
                display: flex;
                flex-wrap: wrap;
                gap: 0.45rem;
                margin-bottom: 0.65rem;
            }}

            .source-badge {{
                padding: 0.28rem 0.62rem;
                border-radius: 999px;
                font-size: 0.76rem;
                font-weight: 700;
                display: inline-block;
            }}

            .source-budget {{
                background: {badge_budget_bg};
                color: {badge_budget_text};
                border: 1px solid rgba(6,182,212,0.22);
            }}

            .source-election {{
                background: {badge_election_bg};
                color: {badge_election_text};
                border: 1px solid rgba(124,58,237,0.22);
            }}

            .source-mixed {{
                background: {badge_mixed_bg};
                color: {badge_mixed_text};
                border: 1px solid rgba(16,185,129,0.22);
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


def _user_safe_error(prefix: str, err: Exception) -> str:
    detail = str(err).strip()
    if detail:
        return f"{prefix} Please try again. Details: {detail}"
    return f"{prefix} Please try again."


def _query_type_label(qt: str) -> str:
    return {
        "election": "Election data",
        "budget": "Budget PDF",
        "mixed": "Mixed sources",
    }.get(qt, qt.replace("_", " ").title())


def _source_badges_html(result: dict[str, Any]) -> str:
    chunks = result.get("retrieved_chunks", [])
    sources = {str(c.get("source", "")).lower() for c in chunks}

    badges: list[str] = []

    if any("budget" in s or "pdf" in s for s in sources):
        badges.append('<span class="source-badge source-budget">Budget PDF</span>')
    if any("election" in s or "csv" in s for s in sources):
        badges.append('<span class="source-badge source-election">Election CSV</span>')
    if result.get("query_type") == "mixed":
        badges.append('<span class="source-badge source-mixed">Mixed Retrieval</span>')

    if not badges:
        # fallback from query_type
        qt = result.get("query_type")
        if qt == "budget":
            badges.append('<span class="source-badge source-budget">Budget PDF</span>')
        elif qt == "election":
            badges.append('<span class="source-badge source-election">Election CSV</span>')
        else:
            badges.append('<span class="source-badge source-mixed">Mixed Retrieval</span>')

    return f'<div class="source-badge-row">{"".join(badges)}</div>'


_MAX_SAVED_CHATS = 50


def _init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "dark_ui" not in st.session_state:
        st.session_state.dark_ui = False
    if "chat_archives" not in st.session_state:
        st.session_state.chat_archives = []
    if "typing_speed" not in st.session_state:
        st.session_state.typing_speed = 0.008


def _first_user_title(messages: list[dict[str, Any]]) -> str:
    for m in messages:
        if m.get("role") == "user" and (m.get("content") or "").strip():
            t = str(m["content"]).strip()
            return (t[:56] + "…") if len(t) > 56 else t
    return "Conversation"


def _archive_current_thread() -> None:
    """Push the active thread into Past conversations (before clearing)."""
    msgs = st.session_state.get("messages") or []
    if not msgs:
        return
    st.session_state.chat_archives.insert(
        0,
        {
            "title": _first_user_title(msgs),
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "message_count": len(msgs),
            "messages": copy.deepcopy(msgs),
        },
    )
    st.session_state.chat_archives = st.session_state.chat_archives[:_MAX_SAVED_CHATS]


def _render_sidebar_archives() -> None:
    st.markdown("**Past conversations**")
    arch: list = st.session_state.get("chat_archives") or []
    if not arch:
        st.caption("When you click **New chat**, your current thread is saved here. Open any row to continue.")
        return

    st.caption(f"{len(arch)} saved — newest first")
    for idx, item in enumerate(arch):
        title = str(item.get("title", "Chat"))
        meta = f'{item.get("saved_at", "")} · {item.get("message_count", 0)} messages'
        btn_label = title if len(title) <= 44 else title[:41] + "…"
        if st.button(
            btn_label,
            key=f"open_chat_archive_{idx}",
            use_container_width=True,
            help=f"{title}\n{meta}",
        ):
            st.session_state.messages = copy.deepcopy(item["messages"])
            st.rerun()
        st.caption(meta)

    if arch and st.button("Clear saved chats", use_container_width=True, key="clear_chat_archives"):
        st.session_state.chat_archives = []
        st.rerun()


def _thread_to_markdown(messages: list[dict[str, Any]]) -> str:
    lines = ["# AcaIntel AI — exported chat\n"]
    for m in messages:
        role = m.get("role", "")
        if role == "user":
            lines.append("\n## User\n\n")
            lines.append((m.get("content") or "").strip())
        elif role == "assistant":
            lines.append("\n## Assistant\n\n")
            lines.append((m.get("content") or "").strip())
            r = m.get("result")
            if r and r.get("effective_query"):
                lines.append(f"\n\n_Retrieval query (after rewrite): {r['effective_query']}_")
        lines.append("\n")
    return "\n".join(lines).strip() + "\n"


def _thread_to_json(messages: list[dict[str, Any]]) -> str:
    out: list[dict[str, Any]] = []
    for m in messages:
        row: dict[str, Any] = {"role": m.get("role"), "content": m.get("content")}
        r = m.get("result")
        if r:
            row["query_type"] = r.get("query_type")
            row["effective_query"] = r.get("effective_query")
            row["retrieved_chunk_ids"] = [c["chunk_id"] for c in r.get("retrieved_chunks", [])]
            row["final_answer"] = r.get("response")
        out.append(row)
    return json.dumps(out, indent=2, ensure_ascii=False)


def _hero() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">🧠 AcaIntel AI</div>
            <div class="hero-sub">
                Academic Intelligence Assistant for <strong>Ghana Election Results</strong> and the
                <strong>2025 Budget Statement</strong>, powered by a custom RAG pipeline.
            </div>
            <div class="pill-row">
                <div class="pill">Hybrid Retrieval</div>
                <div class="pill">FAISS + BM25</div>
                <div class="pill">Grounded Answers</div>
                <div class="pill">Evidence View</div>
                <div class="pill">Exam-Ready Demo</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-label">Search Scope</div>
                <div class="stat-value">Election CSV + Budget PDF</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-label">Response Style</div>
                <div class="stat-value">Natural-language answers</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="stat-card">
                <div class="stat-label">Transparency</div>
                <div class="stat-value">Chunks, scores, prompt</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_typing_answer(text: str, *, speed: float) -> None:
    placeholder = st.empty()
    rendered = ""

    safe_text = html.escape(text).replace("\n", "<br>")

    # simple word-based typing animation
    words = safe_text.split(" ")
    for i, word in enumerate(words, start=1):
        rendered = " ".join(words[:i])
        placeholder.markdown(
            f'<div class="assistant-bubble">{rendered}▌</div>',
            unsafe_allow_html=True,
        )
        time.sleep(speed)

    placeholder.markdown(
        f'<div class="assistant-bubble">{safe_text}</div>',
        unsafe_allow_html=True,
    )


def _render_evidence_panel(result: dict[str, Any], *, show_debug: bool) -> None:
    chunks = result["retrieved_chunks"]
    top_score = max((c.get("final_score", 0.0) for c in chunks), default=0.0)

    with st.expander("🔍 View AI reasoning", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Focus", _query_type_label(result["query_type"]))
        c2.metric("Chunks used", len(chunks))
        c3.metric("Top score", f"{top_score:.3f}")

        st.markdown('<div class="section-label">Retrieved evidence</div>', unsafe_allow_html=True)

        if not chunks:
            st.caption("No chunks were retrieved for this query.")
        else:
            for rank, item in enumerate(chunks, start=1):
                label = f"#{rank} · {item['source']} · {item['chunk_id']} · {item['final_score']:.3f}"
                with st.expander(label, expanded=False):
                    s1, s2, s3, s4 = st.columns(4)
                    s1.caption("Type")
                    s1.write(item.get("chunk_type", "—"))
                    s2.caption("Vector")
                    s2.write(f"{item.get('vector_score', 0.0):.3f}")
                    s3.caption("BM25")
                    s3.write(f"{item.get('bm25_score', 0.0):.3f}")
                    s4.caption("Domain Bonus")
                    s4.write(f"{item.get('domain_bonus', 0.0):.3f}")
                    st.markdown("**Chunk text**")
                    st.write(item["text"])

        st.markdown('<div class="section-label">Final prompt</div>', unsafe_allow_html=True)
        st.code(result["final_prompt"], language="text")

        if show_debug:
            st.markdown('<div class="section-label">Debug JSON</div>', unsafe_allow_html=True)
            dbg: dict[str, Any] = {
                "query_type": result["query_type"],
                "effective_query": result.get("effective_query"),
                "chunking_strategy": result.get("chunking_strategy_used"),
                "selected_chunk_ids": [c["chunk_id"] for c in chunks],
                "log_file": str(result.get("log_path")),
            }
            st.json(dbg)


def main() -> None:
    _init_session()
    _inject_styles(dark=st.session_state.dark_ui)

    try:
        pipeline = get_pipeline()
    except Exception as err:
        st.error(
            _user_safe_error(
                "AcaIntel could not start because the data pipeline failed to initialize.",
                err,
            )
        )
        st.info(
            "Check that `data/Ghana_Election_Result.csv` and `data/2025_budget.pdf` exist, then restart the app."
        )
        st.stop()

    llm_offline = getattr(pipeline.llm, "offline", True)
    llm_provider = getattr(pipeline.llm, "provider", "offline")

    with st.sidebar:
        st.markdown('<div class="sidebar-title">🧠 AcaIntel AI</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-sub">Premium demo UI for your custom Academic City RAG assistant.</div>',
            unsafe_allow_html=True,
        )

        st.session_state.dark_ui = st.toggle("Dark interface", value=st.session_state.dark_ui)

        st.divider()

        if st.button("New chat", use_container_width=True, type="primary"):
            _archive_current_thread()
            st.session_state.messages = []
            st.rerun()

        _render_sidebar_archives()

        st.divider()

        st.markdown("**Export current chat**")
        if st.session_state.messages:
            st.download_button(
                label="Download .md",
                data=_thread_to_markdown(st.session_state.messages),
                file_name="acaintel_chat.md",
                mime="text/markdown",
                use_container_width=True,
                key="export_md",
            )
            st.download_button(
                label="Download .json",
                data=_thread_to_json(st.session_state.messages),
                file_name="acaintel_chat.json",
                mime="application/json",
                use_container_width=True,
                key="export_json",
            )
        else:
            st.caption("Send a message to enable export.")

        st.divider()

        st.markdown("**Retrieval settings**")
        top_k = st.slider("Chunks (top-k)", 2, 6, 4)
        prompt_version = st.selectbox(
            "Prompt version",
            ["v1", "v2", "v3"],
            index=2,
            format_func=lambda x: {
                "v1": "v1 Basic",
                "v2": "v2 Strict",
                "v3": "v3 Structured",
            }[x],
        )
        show_debug = st.toggle("Show extra debug JSON", value=False)
        use_api_stream = st.toggle(
            "Stream from API",
            value=False,
            disabled=llm_offline,
            help="Live token stream from the active model provider. Overrides typing animation.",
        )
        typing_effect = st.toggle(
            "Typing animation",
            value=True,
            disabled=use_api_stream and not llm_offline,
            help="Simulated typing when not using API streaming.",
        )
        st.session_state.typing_speed = st.slider("Typing speed", 0.0, 0.03, 0.008, 0.002)

        st.divider()

        st.markdown("**System status**")
        if llm_offline:
            st.markdown('<span class="status-ok">Offline Mode · $0</span>', unsafe_allow_html=True)
        elif llm_provider == "ollama":
            st.markdown(
                f'<span class="status-ok">Ollama Local · {pipeline.llm.model}</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<span class="status-ok">API Connected · {pipeline.llm.model}</span>',
                unsafe_allow_html=True,
            )

        st.divider()
        st.caption("Tip: Use **New chat** to save this thread to Past conversations, then start fresh.")

    _hero()

    for msg in st.session_state.messages:
        role = msg["role"]
        with st.chat_message(role):
            if role == "user":
                st.markdown(
                    f'<div class="user-bubble">{html.escape(msg["content"])}</div>',
                    unsafe_allow_html=True,
                )
            else:
                result = msg.get("result")
                if result is not None:
                    st.markdown(_source_badges_html(result), unsafe_allow_html=True)

                st.markdown(
                    f'<div class="assistant-bubble">{html.escape(msg["content"]).replace(chr(10), "<br>")}</div>',
                    unsafe_allow_html=True,
                )
                if result is not None:
                    _render_evidence_panel(result, show_debug=show_debug)

    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown(
                """
                <div class="assistant-bubble">
                    <strong>Welcome — I’m AcaIntel AI.</strong><br><br>
                    I answer questions using:
                    <ul>
                        <li>Ghana Election Results</li>
                        <li>2025 Budget Statement</li>
                    </ul>
                    I use retrieval first, then generate a grounded answer in natural language.<br><br>
                    <span class="mini-note">
                        Try asking about constituencies, parties, allocations, revenue, debt, inflation, or education spending.
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if prompt := st.chat_input("Ask AcaIntel AI about elections or the 2025 budget…"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            prep = pipeline.prepare_retrieval(
                query=prompt.strip(),
                top_k=top_k,
                prompt_version=prompt_version,
            )
        except Exception as err:
            error_text = _user_safe_error(
                "I could not retrieve evidence for that question.",
                err,
            )
            with st.chat_message("assistant"):
                st.markdown(
                    f'<div class="assistant-bubble">{html.escape(error_text)}</div>',
                    unsafe_allow_html=True,
                )
            st.session_state.messages.append({"role": "assistant", "content": error_text})
            prep = None

        if prep is not None:
            badge_ctx = {
                "query_type": prep["query_type"],
                "retrieved_chunks": prep["retrieved_chunks"],
            }

            with st.chat_message("assistant"):
                st.markdown(_source_badges_html(badge_ctx), unsafe_allow_html=True)

                stream_on = use_api_stream and not llm_offline
                result: dict[str, Any] | None = None

                try:
                    if stream_on:
                        acc: list[str] = []

                        def _stream_gen():
                            for t in pipeline.llm.generate_stream(prep["final_prompt"]):
                                acc.append(t)
                                yield t

                        streamed = st.write_stream(_stream_gen())
                        full_text = streamed if isinstance(streamed, str) and streamed else "".join(acc)
                    else:
                        with st.spinner("Analyzing sources and preparing answer…"):
                            full_text = pipeline.llm.generate(
                                prep["final_prompt"],
                                query=prep["query"],
                                chunks=prep["retrieved_chunks"],
                            )
                        if typing_effect:
                            _render_typing_answer(full_text, speed=st.session_state.typing_speed)
                        else:
                            st.markdown(
                                f'<div class="assistant-bubble">{html.escape(full_text).replace(chr(10), "<br>")}</div>',
                                unsafe_allow_html=True,
                            )

                    result = pipeline.finalize_answer(prep, full_text)
                except Exception as err:
                    error_text = _user_safe_error(
                        "I hit an error while generating the final answer.",
                        err,
                    )
                    st.markdown(
                        f'<div class="assistant-bubble">{html.escape(error_text)}</div>',
                        unsafe_allow_html=True,
                    )
                    result = {"response": error_text}

                if result.get("retrieved_chunks") is not None and result.get("final_prompt") is not None:
                    _render_evidence_panel(result, show_debug=show_debug)

            assistant_msg: dict[str, Any] = {"role": "assistant", "content": result["response"]}
            if "retrieved_chunks" in result:
                assistant_msg["result"] = result
            st.session_state.messages.append(assistant_msg)

    st.markdown(
        """
        <div class="footer-note">
            Built with a custom RAG architecture · grounded responses · evidence-first design
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()