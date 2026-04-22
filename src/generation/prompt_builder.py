from __future__ import annotations

# Exact phrase the model (and offline fallback) must use when context is insufficient.
UNKNOWN_FROM_DOCUMENTS = "I do not have enough information from the provided documents."


def build_context(chunks: list[dict]) -> str:
    parts: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        header = (
            f"[Chunk {idx} | chunk_id={chunk['chunk_id']} | source={chunk['source']} "
            f"| final_score={chunk['final_score']:.3f}]"
        )
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(parts)


def build_prompt(query: str, chunks: list[dict], version: str = "v3") -> str:
    context = build_context(chunks)

    if version == "v1":
        return (
            "You are an AI assistant for Academic City.\n\n"
            "Use ONLY the following context to answer. Write a clear answer in one short paragraph using natural language.\n"
            "Do not paste raw context, chunk headers, or bullet lists of retrieved text into your answer.\n"
            f'If the answer cannot be found in the context, reply exactly with: "{UNKNOWN_FROM_DOCUMENTS}"\n\n'
            f"Context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            "Answer (natural language only):"
        )

    if version == "v2":
        return (
            "You are an AI assistant for Academic City.\n\n"
            "Answer the question ONLY using the provided context.\n"
            "Respond in one short paragraph of natural language. Do not include retrieved text verbatim or chunk labels.\n"
            f'If the answer is not in the context, reply exactly with: "{UNKNOWN_FROM_DOCUMENTS}"\n\n'
            f"Context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            "Answer (natural language only):"
        )

    return (
        "You are an AI assistant for Academic City that answers using ONLY the retrieved context below.\n\n"
        "Rules:\n"
        "- Use ONLY facts supported by the context; do not invent details.\n"
        "- Write ONE short paragraph in plain natural language (no markdown headings, no chunk IDs, no quoted dumps of the context).\n"
        "- Synthesize the answer; do not copy-paste multi-line excerpts from the context into your reply.\n"
        f'- If the context does not contain enough information to answer, reply exactly with: "{UNKNOWN_FROM_DOCUMENTS}"\n'
        "- When giving numbers (e.g. votes), state the figure and what it refers to (e.g. candidate, region, year).\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        "Answer (natural language only):"
    )
