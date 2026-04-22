from __future__ import annotations

import re


def normalize_pdf_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_pdf_pages(pages: list[dict]) -> list[dict]:
    cleaned_pages: list[dict] = []
    for page in pages:
        cleaned_pages.append(
            {
                "page_number": page["page_number"],
                "text": normalize_pdf_text(page["text"]),
            }
        )
    return cleaned_pages
