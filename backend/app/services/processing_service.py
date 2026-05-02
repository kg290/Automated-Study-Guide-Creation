import re
from collections import Counter

from langchain_text_splitters import RecursiveCharacterTextSplitter


def _normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", (line or "").strip())


def _looks_like_file_reference(line: str) -> bool:
    compact = _normalize_line(line)
    if not compact:
        return False

    lowered = compact.lower()

    # Paths and common filename-like tokens.
    if "\\" in compact or "/" in compact:
        if re.search(r"\.[a-z0-9]{1,8}(?:\b|$)", lowered):
            return True

    if re.search(
        r"\b[\w\-. ]+\.(pdf|doc|docx|ppt|pptx|xls|xlsx|csv|txt|md|py|java|js|ts|json|xml|html)\b",
        lowered,
    ):
        return True

    # "source: something.pdf" style metadata lines.
    if re.search(
        r"\b(source|file|filename|document)\s*:\s*.+\.[a-z0-9]{1,8}\b", lowered
    ):
        return True

    return False


def _low_information_ratio(line: str) -> float:
    tokens = re.findall(r"[A-Za-z0-9\-]+", line)
    if not tokens:
        return 1.0

    alpha = [t for t in tokens if re.search(r"[A-Za-z]", t)]
    if not alpha:
        return 0.95

    long_alpha = [t for t in alpha if len(t) >= 3]
    stopish = {
        "page",
        "pages",
        "chapter",
        "unit",
        "source",
        "file",
        "document",
        "index",
        "contents",
    }
    stopish_count = sum(1 for t in alpha if t.lower() in stopish)

    # Higher means less informative.
    short_alpha_ratio = 1.0 - (len(long_alpha) / max(1, len(alpha)))
    stopish_ratio = stopish_count / max(1, len(alpha))
    numeric_ratio = (len(tokens) - len(alpha)) / max(1, len(tokens))

    return min(
        1.0, 0.45 * short_alpha_ratio + 0.35 * stopish_ratio + 0.20 * numeric_ratio
    )


def _is_low_information_line(line: str) -> bool:
    compact = _normalize_line(line)
    if not compact:
        return False

    # Pure page numbers or roman numerals.
    if re.fullmatch(r"(page\s*)?\d{1,4}", compact, flags=re.IGNORECASE):
        return True
    if re.fullmatch(r"[ivxlcdm]{1,10}", compact, flags=re.IGNORECASE):
        return True

    # Very short heading-like noise.
    if len(compact) <= 4:
        return True

    # Table-of-contents / dotted leaders style.
    if re.search(r"\.{3,}\s*\d{1,4}$", compact):
        return True

    # Repeated single-letter token pattern => OCR noise.
    if re.search(r"(?:\b[a-zA-Z]\b\s+){6,}", compact):
        return True

    # Weak lexical content.
    if _low_information_ratio(compact) >= 0.72 and len(compact) < 80:
        return True

    return False


def _looks_like_code_line(line: str) -> bool:
    compact = _normalize_line(line)
    if not compact:
        return False

    code_keywords = (
        "class ",
        "public:",
        "private:",
        "protected:",
        "return ",
        "void ",
        "int ",
        "bool ",
        "vector",
        "string ",
        "nullptr",
        "push(",
        "pop(",
        "size()",
        "for(",
        "while(",
        "if(",
        "else",
        "node->",
        "::",
    )
    signal_count = sum(1 for token in code_keywords if token in compact)
    punctuation_signals = sum(compact.count(symbol) for symbol in (";", "{", "}", "->", "::"))
    bracket_pairs = compact.count("(") + compact.count(")") + compact.count("[") + compact.count("]")

    natural_words = re.findall(r"[A-Za-z]{3,}", compact)
    identifier_words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", compact)
    natural_ratio = len(natural_words) / max(1, len(identifier_words))

    if signal_count >= 2:
        return True
    if punctuation_signals >= 3 and bracket_pairs >= 2:
        return True
    if compact.count("=") >= 2 and bracket_pairs >= 2:
        return True
    if natural_ratio < 0.34 and (punctuation_signals >= 2 or bracket_pairs >= 4):
        return True

    return False


def clean_extracted_text(raw_text: str) -> str:
    if not raw_text:
        return ""

    normalized = raw_text.replace("\u00a0", " ").replace("\t", " ")
    lines = [line.strip() for line in normalized.splitlines()]

    # Remove highly repeated boilerplate lines (headers/footers).
    candidate_lines = [
        _normalize_line(line)
        for line in lines
        if 5 <= len(_normalize_line(line)) <= 140
    ]
    frequency = Counter(candidate_lines)
    repeated_lines = {line for line, count in frequency.items() if count >= 3}

    cleaned_lines: list[str] = []
    for raw_line in lines:
        compact_line = _normalize_line(raw_line)

        if not compact_line:
            cleaned_lines.append("")
            continue

        if compact_line in repeated_lines:
            continue

        if _looks_like_file_reference(compact_line):
            continue

        if _is_low_information_line(compact_line):
            continue

        if _looks_like_code_line(compact_line):
            continue

        # Keep most punctuation useful for semantics, remove odd OCR artifacts.
        compact_line = re.sub(r"[^\w\s\.,;:!\?\-\(\)\[\]/%+=]", "", compact_line)
        compact_line = compact_line.replace("**", "").replace("`", "")
        cleaned_lines.append(compact_line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)

    return text.strip()


def split_into_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return [chunk for chunk in splitter.split_text(text) if chunk.strip()]
