from pathlib import Path
from pypdf import PdfReader

from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def load_pdf(file_path: str) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(file_path)

    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)

    return "\n".join(pages)


def load_text(file_path: str) -> str:
    """Read text from a TXT file."""
    return Path(file_path).read_text(
        encoding="utf-8"
    )


def load_documents(folder_path: str = "data/documents"):
    """Load PDF and TXT documents from a folder."""
    folder = Path(folder_path)

    documents = []

    for file_path in folder.iterdir():

        if file_path.suffix.lower() == ".pdf":
            text = load_pdf(str(file_path))

        elif file_path.suffix.lower() == ".txt":
            text = load_text(str(file_path))

        else:
            continue

        documents.append({
            "source": file_path.name,
            "text": text
        })

    return documents


def chunk_text(text: str):
    """Split text into overlapping chunks."""
    chunks = []

    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks