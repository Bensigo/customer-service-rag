"""Unit specs for the LLM client adapters (issue #18).

The adapters wrap LangChain chat models behind a small sync ``complete``
Protocol. These tests never hit Ollama or Anthropic: the LangChain chat
model is a fake with a recording ``invoke``, and the builder tests only
inspect the *constructed* model's config (both ChatOllama and
ChatAnthropic build fully offline).
"""

import pytest

from app.chat.llm import (
    LangChainAnthropicClient,
    LangChainOllamaClient,
    LLMError,
    create_llm_client,
)
from app.config import Settings
from app.models import Message


class FakeChatModel:
    """Stands in for a LangChain chat model: records the messages passed
    to invoke and returns a canned AIMessage-like object."""

    def __init__(self, *, reply="canned answer", raises=None):
        self.reply = reply
        self.raises = raises
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.raises is not None:
            raise self.raises
        return _FakeAIMessage(self.reply)


class _FakeAIMessage:
    def __init__(self, text):
        self._text = text
        self.content = text

    @property
    def text(self):
        return self._text


def _messages():
    return [
        Message(role="system", content="you are support"),
        Message(role="user", content="hi"),
        Message(role="assistant", content="hello"),
        Message(role="user", content="reset my password"),
    ]


def test_ollama_client_returns_invoke_text():
    fake = FakeChatModel(reply="here is how to reset")
    client = LangChainOllamaClient(fake)

    answer = client.complete(_messages())

    assert answer == "here is how to reset"


def test_ollama_client_maps_roles_to_langchain_messages():
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    fake = FakeChatModel()
    client = LangChainOllamaClient(fake)

    client.complete(_messages())

    sent = fake.calls[0]
    assert [type(m) for m in sent] == [
        SystemMessage,
        HumanMessage,
        AIMessage,
        HumanMessage,
    ]
    assert sent[0].content == "you are support"
    assert sent[3].content == "reset my password"


def test_client_wraps_invoke_failure_as_llm_error():
    fake = FakeChatModel(raises=RuntimeError("ollama down"))
    client = LangChainOllamaClient(fake)

    with pytest.raises(LLMError):
        client.complete(_messages())


def test_llm_error_message_has_no_prompt_content():
    fake = FakeChatModel(raises=RuntimeError("boom"))
    client = LangChainOllamaClient(fake)

    with pytest.raises(LLMError) as excinfo:
        client.complete([Message(role="user", content="SECRET-CUSTOMER-PII-4111111111111111")])

    assert "4111111111111111" not in str(excinfo.value)
    assert "SECRET-CUSTOMER-PII" not in str(excinfo.value)


def test_create_llm_client_defaults_to_ollama_with_reasoning_disabled():
    settings = Settings()  # default provider = ollama, no key needed
    client = create_llm_client(settings)

    assert isinstance(client, LangChainOllamaClient)
    model = client._model
    # reasoning=False -> ChatOllama sends think:false (gemma4 is a thinking
    # model; without this the tiny answer can come back empty)
    assert model.reasoning is False
    assert model.model == "gemma4"
    assert model.base_url == "http://localhost:11434"


def test_create_llm_client_anthropic_omits_sampling_params():
    settings = Settings(llm_provider="anthropic", anthropic_api_key="sk-dummy")
    client = create_llm_client(settings)

    assert isinstance(client, LangChainAnthropicClient)
    model = client._model
    # Sonnet 5 rejects non-default sampling params with a 400 — the adapter
    # must never set temperature/top_p/top_k (they stay None => not sent).
    assert model.temperature is None
    assert model.top_p is None
    assert model.top_k is None
    assert model.model == "claude-sonnet-5"


def test_anthropic_client_returns_invoke_text():
    fake = FakeChatModel(reply="claude answer")
    client = LangChainAnthropicClient(fake)

    assert client.complete(_messages()) == "claude answer"


def test_close_closes_the_underlying_model_client():
    class _ClosableClient:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class _ModelWithClient(FakeChatModel):
        def __init__(self):
            super().__init__()
            self._client = _ClosableClient()

    model = _ModelWithClient()
    client = LangChainOllamaClient(model)

    client.close()

    assert model._client.closed


def test_close_is_a_noop_when_model_has_no_client():
    # A bare fake (no _client) must not raise on close.
    LangChainOllamaClient(FakeChatModel()).close()
