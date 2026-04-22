from src.preprocessing.chunking import extract_keywords, split_paragraphs


def test_extract_keywords_returns_list() -> None:
    words = extract_keywords("Ghana budget allocation education ministry revenue debt.")
    assert isinstance(words, list)
    assert len(words) > 0


def test_split_paragraphs() -> None:
    text = "Paragraph one with enough words for the test case. It has some extra words.\n\nParagraph two also contains enough words for the function to keep it and not discard it."
    paragraphs = split_paragraphs(text)
    assert isinstance(paragraphs, list)
