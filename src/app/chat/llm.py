"""LLM generation client behind a small provider-agnostic seam (issue #18).

The chat endpoint needs one thing from an LLM: turn an assembled prompt
(our ``Message`` list) into an answer string. ``LLMClient`` is that seam
— a single sync ``complete`` — so the endpoint offloads it via
``anyio.to_thread`` exactly like the embedder and reranker, and tests
swap in a ``FakeLLMClient`` with no network.

Two adapters wrap LangChain chat models behind the seam:

- ``LangChainOllamaClient`` — the default (Ollama pivot: zero API keys).
  Built on ``langchain_ollama.ChatOllama`` with ``reasoning=False`` so a
  thinking model (gemma4) spends its budget on the answer, not hidden
  reasoning it never returns — the same gotcha the reranker handles with
  ``think: false``.
- ``LangChainAnthropicClient`` — the documented bring-your-own-key path.
  Built on ``langchain_anthropic.ChatAnthropic``. It must NOT set
  temperature/top_p/top_k: claude-sonnet-5 rejects non-default sampling
  params with a 400, so the common ``temperature=0`` RAG habit breaks
  here. Leaving them unset (None) means LangChain never sends them.

Both models carry a timeout and a bounded retry in their own config, so
``complete`` stays a thin map-invoke-unwrap call. Failures surface as
``LLMError`` whose message never echoes prompt content (chunk text and
user messages are customer data, per CLAUDE.md).
"""

from typing import Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.config import Settings
from app.models import Message

# Generation is a couple of sentences of grounded support text; cap it so a
# runaway model can't stream forever, but leave ample room for a real answer.
_MAX_OUTPUT_TOKENS = 1024
_REQUEST_TIMEOUT_SECONDS = 60.0
_MAX_RETRIES = 2

_ROLE_TO_MESSAGE = {
    "system": SystemMessage,
    "user": HumanMessage,
    "assistant": AIMessage,
}


class LLMError(Exception):
    """Generation failed. Messages never include prompt or answer text."""


class LLMClient(Protocol):
    """The generation seam: assembled messages in, answer text out.

    Synchronous and blocking by design — the chat endpoint offloads it
    via ``anyio.to_thread``, consistent with the embedder and reranker.
    ``close`` releases any pooled HTTP client at shutdown.
    """

    def complete(self, messages: list[Message]) -> str: ...

    def close(self) -> None: ...


class SupportsInvoke(Protocol):
    """A LangChain chat model seam: ``invoke`` a message list, get a
    message whose ``.text`` is the answer. Lets tests inject a fake."""

    def invoke(self, messages: list[BaseMessage]) -> BaseMessage: ...


def _to_langchain(messages: list[Message]) -> list[BaseMessage]:
    """Map our role-tagged messages onto LangChain's message classes."""
    return [_ROLE_TO_MESSAGE[m.role](content=m.content) for m in messages]


class _LangChainClient:
    """Shared adapter: map our messages, invoke the model, return its text.

    Any invoke failure becomes an ``LLMError`` carrying only the error
    type name — never the prompt (which holds untrusted chunk text and
    the customer's message).
    """

    def __init__(self, model: SupportsInvoke) -> None:
        self._model = model

    def complete(self, messages: list[Message]) -> str:
        try:
            response = self._model.invoke(_to_langchain(messages))
        except Exception as error:
            raise LLMError(f"LLM generation failed: {type(error).__name__}") from error
        # ``.text`` is a property on modern LangChain messages; it flattens
        # string or content-block responses to plain text.
        return response.text

    def close(self) -> None:
        """Release the underlying model's pooled HTTP client, if any.

        LangChain's ChatOllama/ChatAnthropic hold an httpx client that
        must be closed on shutdown or the socket leaks. Best-effort: a
        model without a closable client (e.g. a test fake) is a no-op.
        """
        client = getattr(self._model, "_client", None)
        close = getattr(client, "close", None)
        if callable(close):
            close()


class LangChainOllamaClient(_LangChainClient):
    """Default generation client over Ollama's ChatOllama."""


class LangChainAnthropicClient(_LangChainClient):
    """Generation client over Anthropic's ChatAnthropic (BYO key path)."""


def _build_ollama_model(settings: Settings) -> SupportsInvoke:
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        reasoning=False,  # -> think:false; gemma4 must answer, not hide-think
        num_predict=_MAX_OUTPUT_TOKENS,
        client_kwargs={"timeout": _REQUEST_TIMEOUT_SECONDS},
    )


def _build_anthropic_model(settings: Settings) -> SupportsInvoke:
    from langchain_anthropic import ChatAnthropic

    key = settings.anthropic_api_key
    if key is None:  # get_settings() enforces this; guard defensively too
        raise LLMError("ANTHROPIC_API_KEY is required when llm_provider=anthropic")
    # Deliberately no temperature/top_p/top_k: claude-sonnet-5 400s on
    # non-default sampling params. max_tokens + timeout + retry only.
    return ChatAnthropic(
        model=settings.anthropic_model,
        api_key=key.get_secret_value(),
        max_tokens=_MAX_OUTPUT_TOKENS,
        default_request_timeout=_REQUEST_TIMEOUT_SECONDS,
        max_retries=_MAX_RETRIES,
    )


def create_llm_client(settings: Settings) -> LLMClient:
    """Build the generation client for the configured provider.

    Defaults to Ollama (no API key). ``anthropic`` requires a key, which
    Settings already validates at load time.
    """
    if settings.llm_provider == "anthropic":
        return LangChainAnthropicClient(_build_anthropic_model(settings))
    return LangChainOllamaClient(_build_ollama_model(settings))
