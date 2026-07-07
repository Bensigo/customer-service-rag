import traceback
from pathlib import Path

import pytest

from app.ingestion.extractors import (
    EmptyDocumentError,
    ExtractionError,
    UnsupportedFileType,
    extract,
)
from app.models import ExtractedDocument

FIXTURES = Path(__file__).parent / "fixtures"


def fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def test_extract_txt_and_md_returns_text_and_title():
    txt = fixture_bytes("sample.txt")
    doc = extract("sample.txt", txt)
    assert isinstance(doc, ExtractedDocument)
    assert doc.title == "sample"  # .txt title is the filename stem
    assert doc.text == txt.decode("utf-8")

    md = fixture_bytes("sample.md")
    doc = extract("sample.md", md)
    assert doc.title == "Refund Policy"
    assert doc.text == md.decode("utf-8")  # full markdown source, heading line included


def test_md_title_from_first_heading():
    content = b"intro paragraph\n## not this subheading\n# Actual Title\nbody text\n"

    doc = extract("guide.md", content)

    assert doc.title == "Actual Title"


def test_md_without_heading_falls_back_to_filename_stem():
    doc = extract("escalation-runbook.md", b"Just paragraphs. No headings here.\n")

    assert doc.title == "escalation-runbook"


def test_extension_dispatch_is_case_insensitive():
    assert extract("NOTES.TXT", b"upper case extension\n").title == "NOTES"
    assert extract("Guide.MD", b"# Mixed Case\nbody\n").title == "Mixed Case"
    assert extract("REPORT.PDF", fixture_bytes("sample.pdf")).title == "Support Guide"


def test_txt_decoding_falls_back_from_utf8_to_latin1():
    # utf-8-sig first: a UTF-8 BOM is stripped, not leaked into text or title
    bom_md = b"\xef\xbb\xbf# Caf\xc3\xa9 Guide\nespresso machine resets\n"
    doc = extract("cafe.md", bom_md)
    assert doc.title == "Café Guide"
    assert doc.text.startswith("# Café Guide")

    # invalid UTF-8 falls back to latin-1, which maps every byte
    doc = extract("menu.txt", b"caf\xe9 menu\n")
    assert doc.text == "café menu\n"


def test_extract_pdf_returns_page_text():
    doc = extract("sample.pdf", fixture_bytes("sample.pdf"))

    # pages joined with a form-feed; title from PDF metadata
    assert doc.text == "Alpha support notes for agents.\fBravo escalation steps for tier two."
    assert doc.title == "Support Guide"


def test_pdf_without_metadata_title_falls_back_to_filename_stem():
    doc = extract("retention.pdf", fixture_bytes("untitled.pdf"))

    assert doc.title == "retention"
    assert doc.text == "Charlie retention playbook."


def test_unsupported_extension_raises_UnsupportedFileType():
    with pytest.raises(UnsupportedFileType):
        extract("slides.docx", b"binary blob")
    with pytest.raises(UnsupportedFileType):
        extract("README", b"no extension at all")


def test_corrupt_pdf_raises_ExtractionError():
    with pytest.raises(ExtractionError) as excinfo:
        extract("corrupt.pdf", fixture_bytes("corrupt.pdf"))

    # a typed error naming the file, never a raw pypdf internal exception
    assert "corrupt.pdf" in str(excinfo.value)


def test_empty_file_raises_EmptyDocumentError():
    with pytest.raises(EmptyDocumentError):
        extract("empty.txt", b"")
    with pytest.raises(EmptyDocumentError):
        extract("blank.md", b"  \n\t \n")


def test_error_messages_never_include_file_content():
    # uploads may carry customer PII: errors report filename and sizes only
    with pytest.raises(UnsupportedFileType) as excinfo:
        extract("notes.rtf", b"customer SSN 000-11-2222")
    assert "SSN" not in str(excinfo.value)

    corrupt = fixture_bytes("corrupt.pdf")
    assert b"SECRET-FIXTURE-MARKER" in corrupt  # the marker really is in the fixture
    with pytest.raises(ExtractionError) as excinfo:
        extract("corrupt.pdf", corrupt)
    # the WHOLE formatted exception — message, cause chain, frames — must be
    # content-free: pypdf parse errors can embed raw stream bytes in their
    # messages, so the cause chain must be severed, keeping only the class name
    formatted = "".join(traceback.format_exception(excinfo.value))
    assert "SECRET-FIXTURE-MARKER" not in formatted
    assert "pypdf" not in formatted  # no pypdf frames or messages in the chain
