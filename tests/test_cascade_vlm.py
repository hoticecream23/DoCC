import pytest

from vlm_stub import chat, serve

SCHEMA = {"type": "object"}
MESSAGES = [{"role": "user", "content": "hi"}]


def _settings(url, **kw):
    from cascade.config import VLMSettings

    base = dict(base_url=url, model="m", timeout_s=2, retries=2, backoff_s=0.5, max_tokens=100)
    base.update(kw)
    return VLMSettings(**base)


def _client(url, slept=None, **kw):
    from cascade.vlm import VLMClient

    return VLMClient(_settings(url, **kw), sleep=(slept.append if slept is not None else lambda s: None))


def test_returns_content_and_sends_a_deterministic_schema_bound_request():
    with serve([(200, chat('{"a": 1}', model="served-model"), 0, None)]) as (url, reqs):
        reply = _client(url).complete(MESSAGES, SCHEMA)
    assert (reply.content, reply.model, reply.finish_reason) == ('{"a": 1}', "served-model", "stop")
    body = reqs[0]["body"]
    assert reqs[0]["path"] == "/v1/chat/completions"
    assert body["model"] == "m" and body["temperature"] == 0 and body["max_tokens"] == 100
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA


def test_retries_a_503_with_backoff_then_succeeds():
    slept = []
    with serve([(503, "{}", 0, None), (503, "{}", 0, None), (200, chat("{}"), 0, None)]) as (url, reqs):
        reply = _client(url, slept).complete(MESSAGES, SCHEMA)
    assert reply.content == "{}"
    assert len(reqs) == 3
    assert slept == [0.5, 1.0]


def test_gives_up_after_the_configured_retries():
    from cascade.vlm import VLMError

    with serve([(500, "{}", 0, None)]) as (url, reqs):
        with pytest.raises(VLMError, match="gave up after 3 attempts"):
            _client(url).complete(MESSAGES, SCHEMA)
    assert len(reqs) == 3


def test_a_400_is_not_retried():
    from cascade.vlm import VLMError

    with serve([(400, '{"error": "bad"}', 0, None)]) as (url, reqs):
        with pytest.raises(VLMError, match="HTTP 400"):
            _client(url).complete(MESSAGES, SCHEMA)
    assert len(reqs) == 1


def test_a_redirect_is_never_followed():
    from cascade.vlm import VLMError

    # Create a stub that should not be reached (the redirect target)
    with serve([(200, chat("{}"), 0, None)]) as (target_url, target_reqs):
        # Create a stub that redirects to target_url
        redirect_location = target_url + "/chat/completions"
        redirect = (302, "{}", 0, {"Location": redirect_location})
        with serve([redirect]) as (url, reqs):
            with pytest.raises(VLMError, match="HTTP 302"):
                _client(url).complete(MESSAGES, SCHEMA)
            # Verify redirect source got the request
            assert len(reqs) == 1
            # Verify redirect target was NOT reached (redirect was not followed)
            assert len(target_reqs) == 0


def test_a_timeout_is_retried_then_reported():
    from cascade.vlm import VLMError

    with serve([(200, chat("{}"), 1.0, None)]) as (url, reqs):
        with pytest.raises(VLMError, match="gave up"):
            _client(url, timeout_s=0.2, retries=1).complete(MESSAGES, SCHEMA)
    assert len(reqs) == 2


def test_an_unexpected_response_shape_is_an_error():
    from cascade.vlm import VLMError

    with serve([(200, {"choices": []}, 0, None)]) as (url, _):
        with pytest.raises(VLMError, match="unexpected response shape"):
            _client(url).complete(MESSAGES, SCHEMA)


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:8000/v1", "http://localhost:11434/v1", "http://[::1]:8000/v1",
])
def test_loopback_hosts_are_accepted(url):
    from cascade.vlm import check_local

    check_local(url, [])


@pytest.mark.parametrize("url", [
    "http://10.0.0.5:8000/v1", "https://api.example.com/v1", "file:///etc/passwd", "http:///v1",
    "http://evil.com@127.0.0.1/v1", "http://evil.com%2F@127.0.0.1/v1", "http://user:pw@127.0.0.1/v1",
    "http://127.0.0.1:8000/v1?x=1", "http://127.0.0.1:80\t.evil.com/v1",
])
def test_non_local_hosts_are_refused(url):
    from cascade.vlm import check_local

    with pytest.raises(ValueError):
        check_local(url, [])


def test_a_listed_host_is_accepted():
    from cascade.vlm import check_local

    check_local("http://gpu-box.internal:8000/v1", ["gpu-box.internal"])


def test_client_refuses_to_start_without_a_model_or_on_a_remote_host():
    from cascade.vlm import VLMClient

    with pytest.raises(ValueError, match="vlm.model is not set"):
        VLMClient(_settings("http://127.0.0.1:1/v1", model=None))
    with pytest.raises(ValueError, match="not loopback"):
        VLMClient(_settings("http://10.0.0.5:8000/v1"))


def test_proxy_environment_variables_are_ignored(monkeypatch):
    # Create a target server that returns a valid response
    with serve([(200, chat("{}"), 0, None)]) as (target_url, target_reqs):
        # Create a proxy server (shouldn't receive any requests)
        with serve([(200, chat("{}"), 0, None)]) as (proxy_url, proxy_reqs):
            # Extract just the origin (scheme://host:port) without /v1
            proxy_origin = proxy_url.rsplit("/", 1)[0]

            # Set proxy environment variables
            monkeypatch.setenv("HTTP_PROXY", proxy_origin)
            monkeypatch.setenv("http_proxy", proxy_origin)
            monkeypatch.delenv("NO_PROXY", raising=False)
            monkeypatch.delenv("no_proxy", raising=False)

            # Make request to target
            client = _client(target_url)
            reply = client.complete(MESSAGES, SCHEMA)

            # Verify response is valid
            assert reply.content == "{}"
            # Verify target got the request and proxy didn't
            assert len(target_reqs) == 1
            assert len(proxy_reqs) == 0


def test_api_key_is_read_from_the_named_environment_variable(monkeypatch):
    from cascade.vlm import VLMClient

    monkeypatch.delenv("CASCADE_TEST_KEY", raising=False)
    with pytest.raises(ValueError, match="CASCADE_TEST_KEY"):
        VLMClient(_settings("http://127.0.0.1:1/v1", api_key_env="CASCADE_TEST_KEY"))
    monkeypatch.setenv("CASCADE_TEST_KEY", "secret")
    with serve([(200, chat("{}"), 0, None)]) as (url, reqs):
        _client(url, api_key_env="CASCADE_TEST_KEY").complete(MESSAGES, SCHEMA)
    assert reqs[0]["headers"]["Authorization"] == "Bearer secret"
