from __future__ import annotations

import copy
import html
import json
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


MAX_SAVED_CHATS = 50

SUGGESTION_GROUPS: dict[str, list[str]] = {
    "Election": [
        "Which party had the highest votes in Central Region?",
        "What are the NPP votes by region?",
        "Compare vote counts between North East and Savannah regions.",
        "Who won more votes in Upper East Region?",
    ],
    "Budget": [
        "What does the 2025 budget say about inflation?",
        "Summarize one key policy objective in the budget.",
        "What does the budget state about debt service?",
        "What is the education allocation focus in the budget?",
    ],
    "Mixed": [
        "Give one election statistic and one budget economic indicator.",
        "Compare political mandate signals with budget policy targets.",
        "How does regional voting compare with budget priorities?",
        "What are two insights combining election data and budget policy?",
    ],
}


@st.cache_resource(show_spinner=False)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


def init_session() -> None:
    defaults = {
        "messages": [],
        "chat_archives": [],
        "dark_ui": False,
        "suggestion_group_index": 0,
        "response_mode": "detailed",
        "show_ab_panel": False,
        "conversation_summary": "",
        "ollama_target_mode": "local",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def inject_styles(dark: bool) -> None:
    if dark:
        bg = "#0B1020"
        surface = "#121A2B"
        surface_2 = "#182338"
        text = "#F8FAFC"
        muted = "#9AA7BD"
        border = "#26324A"
        user_bg = "#2563EB"
        user_text = "#FFFFFF"
        assistant_bg = "#111827"
        accent = "#7C3AED"
        accent_2 = "#06B6D4"
    else:
        bg = "#F5F7FB"
        surface = "#FFFFFF"
        surface_2 = "#EEF2FF"
        text = "#142033"
        muted = "#64748B"
        border = "#DBE3F0"
        user_bg = "#DBEAFE"
        user_text = "#0F172A"
        assistant_bg = "#FFFFFF"
        accent = "#6D28D9"
        accent_2 = "#0891B2"

    st.markdown(
        f"""
        <style>
            .stApp {{
                background:
                    radial-gradient(circle at top left, rgba(124,58,237,0.12), transparent 24%),
                    radial-gradient(circle at top right, rgba(6,182,212,0.10), transparent 20%),
                    {bg};
                color: {text};
            }}

            .block-container {{
                max-width: 84rem;
                padding-top: 1.2rem;
                padding-bottom: 4rem;
            }}

            section[data-testid="stSidebar"] {{
                background: linear-gradient(180deg, {surface}, {surface_2});
                border-right: 1px solid {border};
            }}

            h1, h2, h3, p, label, span, li {{
                color: {text} !important;
            }}

            [data-testid="stCaption"] {{
                color: {muted} !important;
            }}

            .hero {{
                background: linear-gradient(135deg, {surface}, {surface_2});
                border: 1px solid {border};
                border-radius: 24px;
                padding: 1.4rem;
                margin-bottom: 1.2rem;
                box-shadow: 0 12px 32px rgba(0,0,0,0.08);
            }}

            .hero-title {{
                font-size: 2rem;
                font-weight: 850;
                letter-spacing: -0.04em;
                margin-bottom: 0.25rem;
            }}

            .hero-subtitle {{
                color: {muted} !important;
                font-size: 0.98rem;
                line-height: 1.6;
            }}

            .pill-row {{
                display: flex;
                flex-wrap: wrap;
                gap: 0.45rem;
                margin-top: 0.9rem;
            }}

            .pill {{
                border: 1px solid {border};
                background: rgba(124,58,237,0.08);
                padding: 0.35rem 0.7rem;
                border-radius: 999px;
                font-size: 0.78rem;
                font-weight: 700;
            }}

            .chat-row-user {{
                display: flex;
                justify-content: flex-end;
                margin: 0.55rem 0;
            }}

            .chat-row-assistant {{
                display: flex;
                justify-content: flex-start;
                margin: 0.55rem 0;
            }}

            .bubble {{
                max-width: 78%;
                border-radius: 20px;
                padding: 0.9rem 1rem;
                line-height: 1.6;
                box-shadow: 0 8px 24px rgba(0,0,0,0.07);
                border: 1px solid {border};
                font-size: 0.96rem;
            }}

            .user-bubble {{
                background: {user_bg};
                color: {user_text} !important;
                border-bottom-right-radius: 6px;
            }}

            .assistant-bubble {{
                background: {assistant_bg};
                color: {text} !important;
                border-bottom-left-radius: 6px;
            }}

            .source-line {{
                margin-top: 0.55rem;
                padding-top: 0.45rem;
                border-top: 1px solid {border};
                font-size: 0.78rem;
                color: {muted} !important;
            }}

            .suggestion-card {{
                background: {surface};
                border: 1px solid {border};
                border-radius: 18px;
                padding: 1rem;
                margin: 1rem 0;
                box-shadow: 0 8px 24px rgba(0,0,0,0.05);
            }}

            .section-title {{
                font-size: 0.85rem;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: {muted} !important;
                margin-bottom: 0.5rem;
            }}

            .stButton button {{
                border-radius: 12px !important;
                font-weight: 700 !important;
            }}

            .stButton button[kind="primary"] {{
                background: linear-gradient(90deg, {accent}, {accent_2}) !important;
                border: none !important;
                color: white !important;
            }}

            div[data-testid="stExpander"] {{
                border: 1px solid {border} !important;
                border-radius: 16px !important;
                background: {surface} !important;
                overflow: hidden;
            }}

            [data-testid="stChatInput"] {{
                background: {surface} !important;
                border: 1.5px solid {border} !important;
                border-radius: 18px !important;
                box-shadow: 0 10px 30px rgba(0,0,0,0.08);
            }}

            [data-testid="stChatInput"] textarea {{
                color: {text} !important;
                -webkit-text-fill-color: {text} !important;
                caret-color: {accent_2} !important;
                font-weight: 600 !important;
            }}

            .status-badge {{
                display: inline-block;
                padding: 0.35rem 0.7rem;
                border-radius: 999px;
                background: rgba(16,185,129,0.12);
                border: 1px solid rgba(16,185,129,0.28);
                color: #10B981 !important;
                font-weight: 800;
                font-size: 0.78rem;
            }}

            .footer {{
                text-align: center;
                color: {muted} !important;
                margin-top: 2rem;
                font-size: 0.8rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def user_safe_error(prefix: str, err: Exception) -> str:
    detail = str(err).strip()
    return f"{prefix} Details: {detail}" if detail else prefix


def escape_text(text: str) -> str:
    return html.escape(str(text)).replace("\n", "<br>")


def query_type_label(query_type: str) -> str:
    labels = {
        "election": "Election CSV",
        "budget": "Budget PDF",
        "mixed": "Mixed Sources",
    }
    return labels.get(query_type, query_type.replace("_", " ").title())


def first_user_title(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            title = str(message.get("content", "")).strip()
            return title[:56] + "…" if len(title) > 56 else title
    return "Conversation"


def archive_current_thread() -> None:
    messages = st.session_state.get("messages", [])

    if not messages:
        return

    st.session_state.chat_archives.insert(
        0,
        {
            "title": first_user_title(messages),
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "message_count": len(messages),
            "messages": copy.deepcopy(messages),
        },
    )

    st.session_state.chat_archives = st.session_state.chat_archives[:MAX_SAVED_CHATS]


def compress_history_if_needed() -> None:
    messages = st.session_state.get("messages", [])

    if len(messages) < 16:
        return

    summary_parts = []

    for message in messages[-8:]:
        role = "User" if message.get("role") == "user" else "Assistant"
        content = str(message.get("content", "")).replace("\n", " ").strip()

        if content:
            summary_parts.append(f"{role}: {content[:120]}")

    st.session_state.conversation_summary = " | ".join(summary_parts)[:1200]
    st.session_state.messages = messages[-10:]


def thread_to_markdown(messages: list[dict[str, Any]]) -> str:
    lines = ["# AcaIntel AI — Exported Chat\n"]

    for message in messages:
        role = message.get("role", "").title()
        content = str(message.get("content", "")).strip()

        lines.append(f"\n## {role}\n")
        lines.append(content)

        result = message.get("result")
        if result and result.get("effective_query"):
            lines.append(f"\n\n_Retrieval query: {result['effective_query']}_")

    return "\n".join(lines).strip() + "\n"


def thread_to_json(messages: list[dict[str, Any]]) -> str:
    output = []

    for message in messages:
        row = {
            "role": message.get("role"),
            "content": message.get("content"),
        }

        result = message.get("result")
        if result:
            row["query_type"] = result.get("query_type")
            row["effective_query"] = result.get("effective_query")
            row["confidence"] = result.get("confidence")
            row["retrieved_chunk_ids"] = [
                chunk.get("chunk_id") for chunk in result.get("retrieved_chunks", [])
            ]

        output.append(row)

    return json.dumps(output, indent=2, ensure_ascii=False)


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">🧠 AcaIntel AI</div>
            <div class="hero-subtitle">
                A clean academic RAG chatbot for Ghana Election Results and the Ghana 2025 Budget.
                Built with hybrid retrieval, evidence tracking, and explainable answer generation.
            </div>
            <div class="pill-row">
                <div class="pill">FAISS + BM25</div>
                <div class="pill">Grounded Answers</div>
                <div class="pill">CSV + PDF Retrieval</div>
                <div class="pill">Evidence View</div>
                </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_user_bubble(content: str) -> None:
    st.markdown(
        f"""
        <div class="chat-row-user">
            <div class="bubble user-bubble">{escape_text(content)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_source_line(result: dict[str, Any] | None) -> str:
    if not result:
        return ""

    query_type = query_type_label(str(result.get("query_type", "mixed")))
    confidence = str(result.get("confidence", "medium")).lower()
    chunks_count = len(result.get("retrieved_chunks", []))

    return (
        f'<div class="source-line">Sources: {query_type} · '
        f'Chunks: {chunks_count} · Confidence: {confidence}</div>'
    )


def extract_direct_answer(text: str) -> str:
    text = str(text).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for line in lines:
        lower = line.lower()
        if (
            lower.startswith("direct answer:")
            or lower.startswith("answer:")
            or lower.startswith("comparison result:")
            or lower.startswith("definition:")
            or lower.startswith("policy summary:")
        ):
            return line.split(":", 1)[1].strip()

    return text


def render_assistant_bubble(content: str, result: dict[str, Any] | None = None) -> None:
    answer = str(content).strip()

    if result:
        response = str(result.get("response", "")).strip()
        if response and len(answer.split()) <= 8:
            answer = response

    if not answer:
        answer = "I could not generate a final answer for this query."

    st.markdown(
        f"""
        <div class="chat-row-assistant">
            <div class="bubble assistant-bubble">
                {escape_text(answer)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_additional_information(result: dict[str, Any] | None, show_debug: bool = False) -> None:
    if not result:
        return

    chunks = result.get("retrieved_chunks", [])
    top_score = max((float(chunk.get("final_score", 0.0)) for chunk in chunks), default=0.0)

    with st.expander("Additional Information", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Focus", query_type_label(str(result.get("query_type", "mixed"))))
        c2.metric("Chunks used", len(chunks))
        c3.metric("Top score", f"{top_score:.3f}")

        st.markdown("#### Effective Query")
        st.code(str(result.get("effective_query", "—")), language="text")

        st.markdown("#### Confidence")
        st.write(str(result.get("confidence", "medium")).lower())

        st.markdown("#### Retrieved Chunks")

        if not chunks:
            st.info("No chunks were retrieved for this query.")
        else:
            for index, chunk in enumerate(chunks, start=1):
                source = chunk.get("source", "unknown source")
                chunk_id = chunk.get("chunk_id", "unknown chunk")
                score = float(chunk.get("final_score", 0.0))

                with st.expander(f"Chunk {index} · {source} · {chunk_id} · Score: {score:.3f}"):
                    c1, c2, c3, c4 = st.columns(4)

                    c1.caption("Chunk Type")
                    c1.write(chunk.get("chunk_type", "—"))

                    c2.caption("Vector Score")
                    c2.write(f"{float(chunk.get('vector_score', 0.0)):.3f}")

                    c3.caption("BM25 Score")
                    c3.write(f"{float(chunk.get('bm25_score', 0.0)):.3f}")

                    c4.caption("Domain Bonus")
                    c4.write(f"{float(chunk.get('domain_bonus', 0.0)):.3f}")

                    st.markdown("**Chunk Text**")
                    st.write(chunk.get("text", ""))

        st.markdown("#### Final Prompt")
        st.code(str(result.get("final_prompt", "—")), language="text")

        if show_debug:
            st.markdown("#### Debug JSON")
            st.json(
                {
                    "query_type": result.get("query_type"),
                    "effective_query": result.get("effective_query"),
                    "chunking_strategy": result.get("chunking_strategy_used"),
                    "selected_chunk_ids": [
                        chunk.get("chunk_id") for chunk in result.get("retrieved_chunks", [])
                    ],
                    "log_file": str(result.get("log_path", "")),
                }
            )


def render_chat_history(show_debug: bool) -> None:
    for message in st.session_state.messages:
        role = message.get("role")
        content = str(message.get("content", ""))

        if role == "user":
            render_user_bubble(content)

        elif role == "assistant":
            result = message.get("result")
            render_assistant_bubble(content, result)
            render_additional_information(result, show_debug=show_debug)


def render_welcome_message() -> None:
    if st.session_state.messages:
        return

    st.markdown(
        """
        <div class="chat-row-assistant">
            <div class="bubble assistant-bubble">
                <strong>Welcome — I’m AcaIntel AI.</strong><br><br>
                Ask me questions about Ghana Election Results, the 2025 Budget Statement,
                or comparisons between both sources.<br><br>
                I retrieve evidence first, then generate a grounded answer.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_suggested_questions() -> str | None:
    group_names = list(SUGGESTION_GROUPS.keys())
    group_index = st.session_state.suggestion_group_index % len(group_names)
    current_group = group_names[group_index]

    st.markdown('<div class="suggestion-card">', unsafe_allow_html=True)

    header_col, refresh_col = st.columns([4, 1])
    header_col.markdown(
        f'<div class="section-title">Suggested Questions · {current_group}</div>',
        unsafe_allow_html=True,
    )

    if refresh_col.button("Refresh", use_container_width=True):
        st.session_state.suggestion_group_index = (group_index + 1) % len(group_names)
        st.rerun()

    selected_question = None
    questions = SUGGESTION_GROUPS[current_group]
    cols = st.columns(2)

    for index, question in enumerate(questions):
        with cols[index % 2]:
            if st.button(question, key=f"suggestion_{current_group}_{index}", use_container_width=True):
                selected_question = question

    st.markdown("</div>", unsafe_allow_html=True)

    return selected_question


def render_sidebar(pipeline: RAGPipeline) -> dict[str, Any]:
    llm_offline = getattr(pipeline.llm, "offline", True)
    llm_provider = getattr(pipeline.llm, "provider", "offline")
    llm_model = getattr(pipeline.llm, "model", "offline")
    llm_mode = getattr(pipeline.llm, "ollama_mode", "local")

    with st.sidebar:
        st.markdown("## 🧠 AcaIntel AI")
        st.caption("Clean academic RAG assistant interface.")

        st.session_state.dark_ui = st.toggle(
            "Dark interface",
            value=st.session_state.dark_ui,
        )

        st.divider()

        if st.button("New Chat", use_container_width=True, type="primary"):
            archive_current_thread()
            st.session_state.messages = []
            st.rerun()

        st.markdown("### Past Conversations")
        archives = st.session_state.chat_archives

        if not archives:
            st.caption("Saved chats will appear here after clicking New Chat.")
        else:
            for index, archive in enumerate(archives):
                title = archive.get("title", "Conversation")
                saved_at = archive.get("saved_at", "")
                count = archive.get("message_count", 0)

                if st.button(title, key=f"archive_{index}", use_container_width=True):
                    st.session_state.messages = copy.deepcopy(archive["messages"])
                    st.rerun()

                st.caption(f"{saved_at} · {count} messages")

            if st.button("Clear Saved Chats", use_container_width=True):
                st.session_state.chat_archives = []
                st.rerun()

        st.divider()

        st.markdown("### Export Current Chat")

        if st.session_state.messages:
            st.download_button(
                "Download Markdown",
                data=thread_to_markdown(st.session_state.messages),
                file_name="acaintel_chat.md",
                mime="text/markdown",
                use_container_width=True,
            )

            st.download_button(
                "Download JSON",
                data=thread_to_json(st.session_state.messages),
                file_name="acaintel_chat.json",
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.caption("Ask a question to enable export.")

        st.divider()

        st.markdown("### Retrieval Settings")

        top_k = st.slider("Chunks to retrieve", 2, 6, 4)

        prompt_version = st.selectbox(
            "Prompt version",
            ["v1", "v2", "v3"],
            index=2,
            format_func=lambda value: {
                "v1": "v1 Basic",
                "v2": "v2 Strict",
                "v3": "v3 Structured",
            }[value],
        )

        st.session_state.response_mode = st.selectbox(
            "Answer mode",
            ["concise", "detailed", "examiner"],
            index=["concise", "detailed", "examiner"].index(st.session_state.response_mode),
        )

        show_debug = st.toggle("Show debug JSON", value=False)

        st.session_state.show_ab_panel = st.toggle(
            "Show RAG vs Pure LLM",
            value=st.session_state.show_ab_panel,
        )

        st.divider()

        st.markdown("### System Status")

        if llm_provider == "ollama" and not llm_offline:
            mode_options = ["local", "cloud"]
            current_mode = st.session_state.get("ollama_target_mode", llm_mode)
            if current_mode not in mode_options:
                current_mode = llm_mode if llm_mode in mode_options else "local"
            selected_mode = st.radio(
                "Ollama target",
                options=mode_options,
                index=mode_options.index(current_mode),
                format_func=lambda m: f"Ollama {m.title()}",
                horizontal=True,
            )
            st.session_state.ollama_target_mode = selected_mode
            if selected_mode != llm_mode:
                pipeline.llm.set_ollama_mode(selected_mode)
                llm_mode = selected_mode
                llm_model = getattr(pipeline.llm, "model", llm_model)

        if llm_offline:
            st.markdown(
                '<span class="status-badge">Offline Mode · $0</span>',
                unsafe_allow_html=True,
            )
        elif llm_provider == "ollama":
            target_label = "Ollama Cloud" if llm_mode == "cloud" else "Ollama Local"
            st.markdown(
                f'<span class="status-badge">Ollama is working · {target_label} · {llm_model}</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<span class="status-badge">API Connected · {llm_model}</span>',
                unsafe_allow_html=True,
            )

        if st.session_state.conversation_summary:
            st.divider()
            st.markdown("### Session Summary")
            st.caption(st.session_state.conversation_summary)

    return {
        "top_k": top_k,
        "prompt_version": prompt_version,
        "show_debug": show_debug,
    }


def handle_query_submission(
    query: str,
    pipeline: RAGPipeline,
    top_k: int,
    prompt_version: str,
    show_debug: bool,
) -> None:
    st.session_state.messages.append({"role": "user", "content": query})
    render_user_bubble(query)

    try:
        prep = pipeline.prepare_retrieval(
            query=query.strip(),
            top_k=top_k,
            prompt_version=prompt_version,
        )

        pipeline.llm.response_mode = st.session_state.response_mode

        with st.spinner("Retrieving evidence and generating answer..."):
            if prep.get("response_override"):
                # Structured deterministic route already produced the final answer.
                full_text = str(prep["response_override"])
            else:
                full_text = pipeline.llm.generate(
                    prep["final_prompt"],
                    query=prep["query"],
                    chunks=prep["retrieved_chunks"],
                )

        result = pipeline.finalize_answer(prep, full_text)

    except Exception as err:
        error_text = user_safe_error(
            "I could not complete that request.",
            err,
        )

        assistant_message = {
            "role": "assistant",
            "content": error_text,
        }

        st.session_state.messages.append(assistant_message)
        render_assistant_bubble(error_text)
        return

    raw_answer = (
        result.get("response")
        or result.get("answer")
        or result.get("final_answer")
        or full_text
    )
    final_answer = extract_direct_answer(raw_answer)

    assistant_message = {
        "role": "assistant",
        "content": final_answer,
        "result": result,
    }

    st.session_state.messages.append(assistant_message)

    render_assistant_bubble(final_answer, result)
    render_additional_information(result, show_debug=show_debug)

    compress_history_if_needed()

    if st.session_state.show_ab_panel:
        with st.expander("RAG vs Pure LLM", expanded=False):
            st.markdown("#### RAG Answer")
            st.write(final_answer)

            st.markdown("#### Pure LLM Answer")
            try:
                pure_answer = pipeline.pure_llm_answer(query)
                st.write(pure_answer)
            except Exception as err:
                st.warning(user_safe_error("Pure LLM comparison failed.", err))


def main() -> None:
    init_session()
    inject_styles(dark=st.session_state.dark_ui)

    try:
        pipeline = get_pipeline()
    except Exception as err:
        st.error(
            user_safe_error(
                "AcaIntel could not start because the RAG pipeline failed to initialize.",
                err,
            )
        )
        st.info(
            "Check that `data/Ghana_Election_Result.csv` and `data/2025_budget.pdf` exist."
        )
        st.stop()

    settings = render_sidebar(pipeline)

    render_hero()
    render_welcome_message()
    render_chat_history(show_debug=settings["show_debug"])

    selected_suggestion = render_suggested_questions()

    typed_query = st.chat_input(
        "Ask about elections, budget policy, or compare both sources..."
    )

    active_query = selected_suggestion or typed_query

    if active_query:
        handle_query_submission(
            query=active_query,
            pipeline=pipeline,
            top_k=settings["top_k"],
            prompt_version=settings["prompt_version"],
            show_debug=settings["show_debug"],
        )

    st.markdown(
        """
        <div class="footer">
            Built with a custom RAG architecture · grounded responses · evidence-first design
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()