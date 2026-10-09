"""Turn an NDA file into numbered clauses. Pure code, no model calls.

Splits on section numbering (1., 1.1, 2(a), Section 3 ...). The definitions
section is kept whole so later prompts can carry it alongside each clause,
which is what lets a clause in §6 be read with the §1 meaning of
"Confidential Information".
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from .models import Clause

_HEADING = re.compile(r"^\s*(?:Section\s+)?(\d+(?:\.\d+)*)[.)]?\s+(.*)$", re.IGNORECASE)
_REFERENCE = re.compile(r"(?:Section|Clause|§)\s*(\d+(?:\.\d+)*)", re.IGNORECASE)
_DEFINITION_WORDS = ("definition", "interpretation", "defined terms")
_WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def read_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if suffix == ".docx":
        with zipfile.ZipFile(path) as archive:
            root = ElementTree.fromstring(archive.read("word/document.xml"))
        paragraphs = []
        for paragraph in root.iter(f"{{{_WORD_NS}}}p"):
            paragraphs.append("".join(t.text or "" for t in paragraph.iter(f"{{{_WORD_NS}}}t")))
        return "\n".join(paragraphs)
    return path.read_text(encoding="utf-8")


_VERBS = {"shall", "may", "must", "will", "is", "are", "means", "do", "does"}


def _looks_like_title(fragment: str) -> bool:
    """'Permitted Disclosure' is a title; 'The Adviser shall not permit contractors' is not."""

    words = fragment.split()
    if not words or len(words) > 6 or len(fragment) >= 60:
        return False
    if any(w.lower() in _VERBS for w in words):
        return False
    capitalised = sum(1 for w in words if w[0].isupper())
    return capitalised >= len(words) - 1


def split_clauses(text: str) -> list[Clause]:
    clauses: list[Clause] = []
    current_id: str | None = None
    heading = ""
    buffer: list[str] = []

    def flush() -> None:
        if current_id is None:
            return
        body = " ".join(line.strip() for line in buffer if line.strip())
        if not body:
            return
        clauses.append(
            Clause(
                id=current_id,
                heading=heading,
                text=body,
                references=sorted(set(_REFERENCE.findall(body)) - {current_id}),
            )
        )

    for line in text.splitlines():
        match = _HEADING.match(line)
        if match:
            flush()
            current_id = match.group(1)
            rest = match.group(2).strip()
            # "3. Permitted Disclosure. The Receiving Party may..." keeps the
            # heading separate from the body it introduces.
            first = rest.split(".")[0]
            if "." in rest and _looks_like_title(first):
                # "3. Permitted Disclosure. The Receiving Party may..." — a short
                # title followed by the body.
                heading, _, body_start = rest.partition(".")
                buffer = [body_start]
            else:
                # No title, the sentence itself is the clause body.
                heading, buffer = "", [rest]
            continue
        if current_id is not None:
            buffer.append(line)
    flush()
    return _unique_ids(clauses)


def _unique_ids(clauses: list[Clause]) -> list[Clause]:
    """Amendments and schedules often number their clauses from 1 again.

    Every later stage looks clauses up by id, so a repeated id would hide the
    earlier clause (the main confidentiality clause behind an amendment's
    clause 2). A repeat keeps its number with an occurrence suffix: 2, 2~2.
    """

    seen: dict[str, int] = {}
    out = []
    for clause in clauses:
        n = seen.get(clause.id, 0) + 1
        seen[clause.id] = n
        out.append(clause if n == 1 else clause.model_copy(update={"id": f"{clause.id}~{n}"}))
    return out


def definitions_text(clauses: list[Clause]) -> str:
    """The definitions section, or the first clause when none is labelled."""

    for clause in clauses:
        if any(word in clause.heading.lower() for word in _DEFINITION_WORDS):
            return clause.text
    return clauses[0].text if clauses else ""


def load(path: Path) -> tuple[str, list[Clause]]:
    text = read_text(path)
    return text, split_clauses(text)
