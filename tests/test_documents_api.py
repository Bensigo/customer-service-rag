"""Unit specs for the document upload API (issue #12).

These run everywhere (no live services): the route's IngestionService is
replaced with a fake via FastAPI's dependency-override mechanism, so the
tests exercise the HTTP boundary — multipart parsing, doc_id derivation
and validation, the size cap, error mapping, and the thread offload —
without touching Ollama, Qdrant, or SQLite.

The size cap and the "errors never echo file content" rule are the
security-critical specs, so they are asserted directly here rather than
left to the integration test.
"""

import threading

import pytest
from fastapi.testclient import TestClient

from app.api.documents import get_ingestion_service
from app.ingestion.pipeline import IngestError, IngestResult
from app.main import create_app


class FakeIngestionService:
    """Records ingest calls and returns a bumped version per doc_id.

    Mirrors IngestionService.ingest's signature and return type without
    any I/O, so the API layer can be tested in isolation.
    """

    def __init__(self):
        self.calls = []
        self._versions = {}
        self.thread_names = []

    def ingest(self, doc_id: str, title: str, text: str) -> IngestResult:
        self.thread_names.append(threading.current_thread().name)
        self.calls.append({"doc_id": doc_id, "title": title, "text": text})
        version = self._versions.get(doc_id, 0) + 1
        self._versions[doc_id] = version
        return IngestResult(doc_id=doc_id, version=version, chunk_count=3)


@pytest.fixture
def fake_service():
    return FakeIngestionService()


def _app_with_fake(service):
    """Build the app with a fake ingestion service, no live resources.

    Pre-stashing ingestion_service on app.state makes the lifespan skip
    building the real ChunkStore/Embedder/Qdrant, so unit tests run with
    no backing services; the dependency override is the seam the route
    actually resolves through.
    """
    app = create_app()
    app.state.ingestion_service = service
    # preset the cap so the lifespan never loads live Settings
    app.state.max_upload_bytes = 5_000_000
    app.dependency_overrides[get_ingestion_service] = lambda: service
    return app


@pytest.fixture
def client(fake_service):
    app = _app_with_fake(fake_service)
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _upload(client, *, filename="faq.txt", content=b"hello world", data=None):
    return client.post(
        "/documents",
        files={"file": (filename, content, "text/plain")},
        data=data or {},
    )


def test_post_document_txt_returns_201_with_version_1(client, fake_service):
    response = _upload(client, filename="password-reset.txt", content=b"reset your password")

    assert response.status_code == 201
    body = response.json()
    assert body == {"doc_id": "password-reset", "version": 1, "chunk_count": 3}
    assert fake_service.calls[0]["doc_id"] == "password-reset"
    assert fake_service.calls[0]["text"] == "reset your password"


def test_post_document_uses_explicit_doc_id_form_field(client, fake_service):
    response = client.post(
        "/documents",
        files={"file": ("Anything At All.txt", b"body", "text/plain")},
        data={"doc_id": "custom-id"},
    )

    assert response.status_code == 201
    assert response.json()["doc_id"] == "custom-id"


def test_put_same_doc_id_returns_version_2(client):
    first = client.put(
        "/documents/faq-billing",
        files={"file": ("faq.txt", b"v1 content", "text/plain")},
    )
    second = client.put(
        "/documents/faq-billing",
        files={"file": ("faq.txt", b"v2 content", "text/plain")},
    )

    assert first.status_code == 200
    assert first.json()["version"] == 1
    assert second.status_code == 200
    assert second.json() == {"doc_id": "faq-billing", "version": 2, "chunk_count": 3}


def test_oversized_upload_returns_413(fake_service):
    app = _app_with_fake(fake_service)
    # shrink the cap so the test payload is small; the lifespan honors a
    # pre-set app.state value
    app.state.max_upload_bytes = 10
    with TestClient(app) as client:
        response = client.post(
            "/documents",
            files={"file": ("big.txt", b"x" * 100, "text/plain")},
        )
    app.dependency_overrides.clear()

    assert response.status_code == 413
    # rejected before the ingest service was ever called
    assert fake_service.calls == []


def test_unsupported_extension_returns_415(client, fake_service):
    response = client.post(
        "/documents",
        files={"file": ("archive.zip", b"PK\x03\x04nope", "application/zip")},
    )

    assert response.status_code == 415
    assert fake_service.calls == []


def test_invalid_doc_id_form_field_returns_422(client, fake_service):
    response = client.post(
        "/documents",
        files={"file": ("faq.txt", b"body", "text/plain")},
        data={"doc_id": "Not Valid!"},
    )

    assert response.status_code == 422
    assert fake_service.calls == []


def test_invalid_doc_id_in_put_path_returns_422(client, fake_service):
    response = client.put(
        "/documents/UPPER_CASE",
        files={"file": ("faq.txt", b"body", "text/plain")},
    )

    assert response.status_code == 422
    assert fake_service.calls == []


def test_doc_id_with_trailing_newline_returns_422(client, fake_service):
    # `$` matches before a terminal newline, so a slug + trailing "\n"
    # would slip past re.match; the id must be rejected outright.
    response = client.post(
        "/documents",
        files={"file": ("faq.txt", b"body", "text/plain")},
        data={"doc_id": "support-faq\n"},
    )

    assert response.status_code == 422
    assert fake_service.calls == []


def test_filename_with_no_sluggable_chars_returns_422(client, fake_service):
    # a filename stem that reduces to an empty/invalid slug must be rejected,
    # not silently coerced into an arbitrary doc_id
    response = client.post(
        "/documents",
        files={"file": ("___.txt", b"body", "text/plain")},
    )

    assert response.status_code == 422
    assert fake_service.calls == []


def test_empty_document_returns_422(client, fake_service):
    response = client.post(
        "/documents",
        files={"file": ("blank.txt", b"   \n  ", "text/plain")},
    )

    assert response.status_code == 422
    assert fake_service.calls == []


def test_ingest_error_returns_422(client):
    class Boom(FakeIngestionService):
        def ingest(self, doc_id, title, text):
            raise IngestError(f"document {doc_id!r} contains no chunkable text")

    app = _app_with_fake(Boom())
    with TestClient(app) as c:
        response = c.post(
            "/documents",
            files={"file": ("faq.txt", b"body", "text/plain")},
        )
    app.dependency_overrides.clear()

    assert response.status_code == 422


def test_error_responses_do_not_echo_file_content():
    secret = b"SUPER-SECRET-CUSTOMER-PII-4111111111111111"

    class Boom(FakeIngestionService):
        def ingest(self, doc_id, title, text):
            raise IngestError("ingest blew up with the raw text: " + text)

    app = _app_with_fake(Boom())
    with TestClient(app, raise_server_exceptions=False) as c:
        response = c.post(
            "/documents",
            files={"file": ("faq.txt", secret, "text/plain")},
        )
    app.dependency_overrides.clear()

    assert secret.decode() not in response.text
    assert "SUPER-SECRET" not in response.text
    assert "4111111111111111" not in response.text


def test_slow_ingest_does_not_block_concurrent_health(client, fake_service):
    release = threading.Event()

    def slow_ingest(doc_id, title, text):
        # blocks until the concurrent /health call has returned
        release.wait(timeout=5)
        return IngestResult(doc_id=doc_id, version=1, chunk_count=1)

    fake_service.ingest = slow_ingest

    upload_result = {}

    def do_upload():
        upload_result["response"] = client.post(
            "/documents",
            files={"file": ("faq.txt", b"body", "text/plain")},
        )

    uploader = threading.Thread(target=do_upload)
    uploader.start()
    try:
        # /health must return promptly even while the ingest thread is parked
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok"}
    finally:
        release.set()
        uploader.join(timeout=5)

    assert upload_result["response"].status_code == 201


def test_ingest_runs_off_the_event_loop_thread(client, fake_service):
    """The ingest call must land on a worker thread, not the request/event
    loop thread — proving the anyio.to_thread offload actually happened."""
    _upload(client)

    assert fake_service.thread_names, "ingest was never called"
    for name in fake_service.thread_names:
        assert name != "MainThread"


def test_windows_filename_is_reduced_to_basename(client, fake_service):
    response = client.post(
        "/documents",
        files={"file": (r"C:\Users\admin\Billing FAQ.txt", b"body", "text/plain")},
    )

    assert response.status_code == 201
    # doc_id derives from the basename stem only, never the path prefix
    assert response.json()["doc_id"] == "billing-faq"
    assert fake_service.calls[0]["title"] == "Billing FAQ"
