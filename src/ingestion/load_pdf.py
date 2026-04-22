from __future__ import annotations

from pathlib import Path

import fitz


def load_pdf_text(path: str | Path) -> list[dict]:
    """Extract text from each page of a PDF using PyMuPDF."""
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    pages: list[dict] = []
    with fitz.open(pdf_path) as doc:
        for index, page in enumerate(doc):
            text = page.get_text("text")
            pages.append(
                {
                    "page_number": index + 1,
                    "text": text,
                }
            )
    return pages
