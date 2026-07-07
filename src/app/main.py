from fastapi import FastAPI


def create_app() -> FastAPI:
    app = FastAPI(title="customer-service-rag")

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness probe: process is up. No dependency checks (readiness is #21)."""
        return {"status": "ok"}

    return app
