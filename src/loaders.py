

"""
Loads raw documents (pdf, docx, txt, md) from a folder and returns
plain text along with basic metadata (source filename).
"""

import os
from typing import List, Dict

from pypdf import PdfReader
import docx


def _load_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)
    return "\n".join(pages)


def _load_docx(path: str) -> str:
    document = docx.Document(path)
    return "\n".join(p.text for p in document.paragraphs if p.text.strip())


def _load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


LOADERS = {
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".txt": _load_text,
    ".md": _load_text,
}


def load_documents(folder: str) -> List[Dict]:
    """
    Reads every supported file in `folder` and returns:
    [{"source": "labor_act.pdf", "text": "..."}]
    """
    docs = []
    if not os.path.isdir(folder):
        return docs

    for filename in sorted(os.listdir(folder)):
        ext = os.path.splitext(filename)[1].lower()
        loader = LOADERS.get(ext)
        if not loader:
            continue

        full_path = os.path.join(folder, filename)
        try:
            text = loader(full_path)
        except Exception as e:
            print(f"[loaders] Failed to read {filename}: {e}")
            continue

        text = text.strip()
        if text:
            docs.append({"source": filename, "text": text})

    return docs
