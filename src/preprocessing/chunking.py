from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


def extract_keywords(text: str, max_keywords: int = 8) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z\-']{2,}", text.lower())
    stopwords = {
        "the", "and", "for", "with", "that", "this", "from", "into", "were", "was",
        "have", "has", "had", "are", "but", "not", "its", "their", "our", "your",
        "all", "you", "use", "using", "only", "will", "can", "also", "been", "than",
        "they", "his", "her", "she", "him", "them", "who", "what", "when", "where",
        "why", "how", "about", "which", "into", "through", "year", "page", "speaker",
        "ghana", "budget"
    }
    freq: dict[str, int] = {}
    for token in tokens:
        if token not in stopwords:
            freq[token] = freq.get(token, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [w for w, _ in ranked[:max_keywords]]


def row_to_text(row: pd.Series) -> str:
    return (
        f"Election record. Year: {row.get('Year', 'Unknown')}. "
        f"Old Region: {row.get('Old Region', 'Unknown')}. "
        f"New Region: {row.get('New Region', 'Unknown')}. "
        f"Code: {row.get('Code', 'Unknown')}. "
        f"Candidate: {row.get('Candidate', 'Unknown')}. "
        f"Party: {row.get('Party', 'Unknown')}. "
        f"Votes: {row.get('Votes', 'Unknown')}. "
        f"Vote Percentage: {row.get('Votes(%)', 'Unknown')}."
    )


def chunk_election_records(df: pd.DataFrame) -> list[dict]:
    chunks: list[dict] = []
    for idx, row in df.reset_index(drop=True).iterrows():
        text = row_to_text(row)
        chunks.append(
            {
                "chunk_id": f"election_{idx:04d}",
                "source": "election_csv",
                "chunk_type": "record",
                "text": text,
                "page_number": None,
                "year": str(row.get("Year", "")),
                "section_title": None,
                "keywords": extract_keywords(text),
                "metadata": {
                    "row_index": idx,
                    "new_region": row.get("New Region"),
                    "candidate": row.get("Candidate"),
                    "party": row.get("Party"),
                },
            }
        )
    return chunks


def _word_windows(words: list[str], chunk_size: int, overlap: int) -> Iterable[list[str]]:
    start = 0
    step = max(chunk_size - overlap, 1)
    while start < len(words):
        yield words[start : start + chunk_size]
        start += step


def chunk_pdf_fixed(pages: list[dict], chunk_size: int = 400, overlap: int = 80) -> list[dict]:
    chunks: list[dict] = []
    for page in pages:
        words = page["text"].split()
        for idx, window in enumerate(_word_windows(words, chunk_size, overlap)):
            text = " ".join(window).strip()
            if len(text.split()) < 60:
                continue
            chunks.append(
                {
                    "chunk_id": f"budget_fixed_p{page['page_number']}_{idx:03d}",
                    "source": "budget_pdf",
                    "chunk_type": "fixed",
                    "text": text,
                    "page_number": page["page_number"],
                    "year": "2025",
                    "section_title": None,
                    "keywords": extract_keywords(text),
                    "metadata": {},
                }
            )
    return chunks


def split_paragraphs(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]
    return [p for p in paras if len(p.split()) >= 20]


def chunk_pdf_paragraph_aware(
    pages: list[dict], min_words: int = 300, max_words: int = 500
) -> list[dict]:
    chunks: list[dict] = []
    carryover = ""
    for page in pages:
        paragraphs = split_paragraphs(page["text"])
        current_parts: list[str] = [carryover] if carryover else []
        current_words = len(carryover.split()) if carryover else 0

        for paragraph in paragraphs:
            paragraph_words = len(paragraph.split())
            if current_words + paragraph_words <= max_words:
                current_parts.append(paragraph)
                current_words += paragraph_words
            else:
                combined = "\n\n".join([part for part in current_parts if part]).strip()
                if combined and len(combined.split()) >= min_words:
                    chunk_id = f"budget_para_p{page['page_number']}_{len(chunks):03d}"
                    chunks.append(
                        {
                            "chunk_id": chunk_id,
                            "source": "budget_pdf",
                            "chunk_type": "paragraph",
                            "text": combined,
                            "page_number": page["page_number"],
                            "year": "2025",
                            "section_title": _guess_section_title(combined),
                            "keywords": extract_keywords(combined),
                            "metadata": {},
                        }
                    )
                    last_sentence = _last_sentence(combined)
                    carryover = last_sentence
                    current_parts = [carryover, paragraph]
                    current_words = len(carryover.split()) + paragraph_words
                else:
                    current_parts.append(paragraph)
                    current_words += paragraph_words

        combined = "\n\n".join([part for part in current_parts if part]).strip()
        if combined and len(combined.split()) >= min_words:
            chunk_id = f"budget_para_p{page['page_number']}_{len(chunks):03d}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source": "budget_pdf",
                    "chunk_type": "paragraph",
                    "text": combined,
                    "page_number": page["page_number"],
                    "year": "2025",
                    "section_title": _guess_section_title(combined),
                    "keywords": extract_keywords(combined),
                    "metadata": {},
                }
            )
            carryover = _last_sentence(combined)
        else:
            carryover = combined
    return chunks


def _guess_section_title(text: str) -> str | None:
    first_line = text.splitlines()[0].strip() if text.splitlines() else ""
    if 3 <= len(first_line.split()) <= 12 and first_line.isupper():
        return first_line
    return None


def _last_sentence(text: str, max_words: int = 25) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if not sentences:
        return ""
    last = sentences[-1]
    words = last.split()
    return " ".join(words[-max_words:])
