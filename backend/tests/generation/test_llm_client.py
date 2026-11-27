import json

import httpx
import pytest
from app.core.config import Settings
from app.generation.errors import (
    LLMConfigError,
    LLMError,
    LLMResponseError,
    LLMTimeoutError,
)
from app.generation.llm_client import (
    FakeLLMClient,
    HostedLLMClient,
    OllamaLLMClient,
    build_llm_client,
)


def _response(content="hello"):
    return {"message": {"content": content}}


def test_ollama_request_shape_and_response():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_response())

    client = OllamaLLMClient(
        base_url="http://local",
        model="m",
        timeout=2,
        retries=0,
        max_output_tokens=27,
        transport=httpx.MockTransport(handler),
    )
    assert client.generate("sys", "usr") == "hello"
    assert seen["url"].endswith("/api/chat")
    assert seen["body"]["stream"] is False
    assert seen["body"]["options"]["num_predict"] == 27


def test_hosted_request_shape_and_auth():
    def handler(request):
        assert request.headers["Authorization"] == "Bearer secret"
        body = json.loads(request.content)
        assert body["max_tokens"] == 24
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = HostedLLMClient(
        base_url="http://local/v1",
        model="m",
        timeout=2,
        retries=0,
        max_output_tokens=24,
        api_key="secret",
        transport=httpx.MockTransport(handler),
    )
    assert client.generate("s", "u") == "ok"


def test_retries_server_error_then_succeeds():
    calls = 0

    def handler(_request):
        nonlocal calls
        calls += 1
        return (
            httpx.Response(503) if calls == 1 else httpx.Response(200, json=_response())
        )

    client = OllamaLLMClient(
        base_url="http://local",
        model="m",
        timeout=2,
        retries=1,
        max_output_tokens=24,
        transport=httpx.MockTransport(handler),
    )
    assert client.generate("s", "u") == "hello"
    assert calls == 2


def test_http_failure_does_not_expose_body():
    client = OllamaLLMClient(
        base_url="http://local",
        model="m",
        timeout=2,
        retries=0,
        max_output_tokens=24,
        transport=httpx.MockTransport(
            lambda _r: httpx.Response(400, text="private-body")
        ),
    )
    with pytest.raises(LLMError) as exc:
        client.generate("s", "u")
    assert "private-body" not in str(exc.value)


def test_malformed_response_is_safe_error():
    client = OllamaLLMClient(
        base_url="http://local",
        model="m",
        timeout=2,
        retries=0,
        max_output_tokens=24,
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, json={})),
    )
    with pytest.raises(LLMResponseError):
        client.generate("s", "u")


def test_timeout_maps_to_timeout_error():
    def handler(_r):
        raise httpx.ReadTimeout("hidden detail")

    client = OllamaLLMClient(
        base_url="http://local",
        model="m",
        timeout=2,
        retries=0,
        max_output_tokens=24,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(LLMTimeoutError):
        client.generate("s", "u")


def test_client_factory_and_fake():
    assert isinstance(
        build_llm_client(Settings(llm_provider="ollama")), OllamaLLMClient
    )
    with pytest.raises(LLMConfigError):
        build_llm_client(Settings(llm_provider="hosted"))
    fake = FakeLLMClient("yes")
    assert fake.generate("system", "question") == "yes"
    assert fake.calls == [("system", "question")]
