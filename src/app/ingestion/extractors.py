"""Text extraction for uploaded documents: .txt, .md, and .pdf.

Pure bytes-in, object-out (no temp files). Dispatch is case-insensitive
on the filename extension.

Decoding for .txt/.md tries utf-8-sig first — it accepts exactly the
same input as plain utf-8 but also strips a leading BOM — then falls
back to latin-1, which maps every byte and so never fails; legacy
8-bit encodings other than latin-1 may misrender, an accepted
trade-off for support documents.

Uploads may carry customer PII, so exception messages name only the
filename and byte size — never file content. Corrupt PDFs surface as
``ExtractionError`` with the short pypdf diagnostic kept on the cause
chain for debugging; a raw pypdf internal exception never escapes.
"""

import warnings
from io import BytesIO
from pathlib import PurePosixPath

from pypdf import PdfReader

from app.models import ExtractedDocument

_PAGE_SEPARATOR = "\f"


class UnsupportedFileType(Exception):
    """The filename extension is not one extraction supports."""


class ExtractionError(Exception):
    """The file matched a supported type but could not be parsed."""


class EmptyDocumentError(Exception):
    """Extraction succeeded but produced no text worth indexing."""


def extract(filename: str, content: bytes) -> ExtractedDocument:
    """Extract title and plain text from an uploaded file's bytes.

    Titles: first markdown ``# `` heading for .md, PDF metadata title
    for .pdf, the filename stem when those are missing (and always for
    .txt). PDF page texts are joined with a form-feed.

    Raises UnsupportedFileType, ExtractionError, or EmptyDocumentError.
    """
    suffix = PurePosixPath(filename).suffix.lower()
    if suffix == ".txt":
        title, text = None, _decode(content)
    elif suffix == ".md":
        text = _decode(content)
        title = _first_markdown_heading(text)
    elif suffix == ".pdf":
        title, text = _extract_pdf(filename, content)
    else:
        raise UnsupportedFileType(
            f"unsupported file type {suffix or '(no extension)'!r}: "
            f"{filename!r} ({len(content)} bytes)"
        )
    if not text.strip():
        raise EmptyDocumentError(f"no text extracted from {filename!r} ({len(content)} bytes)")
    if not (title and title.strip()):
        title = PurePosixPath(filename).stem
    return ExtractedDocument(title=title, text=text)


def _decode(content: bytes) -> str:
    """Decode text bytes: utf-8-sig (plain utf-8 plus BOM stripping),
    then latin-1 as the never-failing last resort."""
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return content.decode("latin-1")


def _first_markdown_heading(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# ") and stripped[2:].strip():
            return stripped[2:].strip()
    return None


def _extract_pdf(filename: str, content: bytes) -> tuple[str | None, str]:
    with warnings.catch_warnings():
        # pypdf reports recoverable corruption as warnings; escalate them
        # so damaged uploads become ExtractionError instead of silently
        # bad text (this also keeps pytest's filterwarnings=error clean)
        warnings.simplefilter("error")
        try:
            reader = PdfReader(BytesIO(content))
            text = _PAGE_SEPARATOR.join(page.extract_text() for page in reader.pages)
            metadata = reader.metadata
            title = None if metadata is None else metadata.title
        except Exception as exc:
            # pypdf raises assorted internal types on corrupt input; map
            # them all to the one typed error, content-free (PII rule)
            raise ExtractionError(
                f"could not extract text from {filename!r} ({len(content)} bytes)"
            ) from exc
    return title, text
