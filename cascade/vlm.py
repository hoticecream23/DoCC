"""Client for a local OpenAI compatible chat completions server.

vLLM serves this on a GPU box, Ollama on a laptop. Document content must not
leave the company, so the client:

- refuses any host that is not loopback or listed in vlm.allowed_hosts;
- never follows a redirect, which could point anywhere;
- ignores proxy environment variables, which could route anywhere.
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import logging
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import IO, Any, Protocol
from urllib.parse import urlparse

from baseline.errors import VLMError

from .config import VLMSettings
from .prompt import ChatMessage

log = logging.getLogger(__name__)

# Code points at or below this are whitespace or C0 control characters.
LAST_CONTROL_CHAR = 32
DEL_CHAR = 127
HTTP_BAD_REQUEST = 400


class VLMBadRequest(VLMError):  # noqa: N818 - established public name, shown in diagnostics
    """HTTP 400. Ollama sends it when the prompt overflows the context."""


@dataclass(frozen=True)
class VLMReply:
    """One answer from the server."""

    content: str
    model: str
    finish_reason: str | None


class Backend(Protocol):
    """Anything that can answer a chat request: the client, or a test fake."""

    def complete(self, messages: list[ChatMessage], schema: dict[str, Any]) -> VLMReply:
        """Send the messages and return the answer, constrained to `schema`."""
        ...


def check_local(url: str, allowed_hosts: list[str]) -> None:
    """Refuse a URL that could send document content off the machine.

    Args:
        url: The server URL to check.
        allowed_hosts: Non loopback hosts the deployment trusts.

    Raises:
        ValueError: The URL is malformed, carries credentials or a query, or
            its host is not loopback and not allowed.
    """
    if any(ord(c) <= LAST_CONTROL_CHAR or ord(c) == DEL_CHAR for c in url):
        raise ValueError(f"VLM url must not contain whitespace or control characters: {url!r}")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"VLM url must be http or https, got {url!r}")

    # Check for userinfo in URL (username:password)
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"VLM url must not contain userinfo: {url!r}")

    # Check for suspicious characters in netloc (%, \, @)
    if any(c in parsed.netloc for c in ["%", "\\", "@"]):
        raise ValueError(f"VLM url contains suspicious characters in netloc: {url!r}")

    # Check for query or fragment
    if parsed.query or parsed.fragment:
        raise ValueError(f"VLM url must not contain query or fragment: {url!r}")

    host = parsed.hostname
    if not host:
        raise ValueError(f"VLM url has no host: {url!r}")
    if host in allowed_hosts or _is_loopback(host):
        return
    raise ValueError(f"VLM host {host!r} is not loopback and not in vlm.allowed_hosts")


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: http.client.HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        return None  # urllib then raises HTTPError with the 3xx status


def _reply(payload: dict[str, Any]) -> VLMReply:
    try:
        choice = payload["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise VLMError(f"unexpected response shape: {exc!r}") from exc
    if not isinstance(content, str):
        raise VLMError("message content is not a string")
    return VLMReply(content=content, model=str(payload.get("model", "")), finish_reason=choice.get("finish_reason"))


class VLMClient:
    """Chat completions client that only talks to a local or allowed host.

    Args:
        settings: Server, model and retry settings.
        sleep: Called between retries; tests pass a no-op.

    Raises:
        ValueError: No model is set, the host is not allowed, or the API key
            variable is empty.
    """

    def __init__(self, settings: VLMSettings, sleep: Callable[[float], None] = time.sleep) -> None:
        if not settings.model:
            raise ValueError("vlm.model is not set: set it in config/cascade.yaml or pass --model")
        check_local(settings.base_url, settings.allowed_hosts)
        self._key: str | None = None
        if settings.api_key_env:
            self._key = os.environ.get(settings.api_key_env)
            if not self._key:
                raise ValueError(f"environment variable {settings.api_key_env} is not set")
            parsed = urlparse(settings.base_url)
            if parsed.scheme != "https" and not _is_loopback(parsed.hostname or ""):
                log.warning("API key sent over plain http to %s; use https off the LAN", parsed.hostname)
        self._s = settings
        self._sleep = sleep
        self._url = settings.base_url.rstrip("/") + "/chat/completions"
        # Validate the final URL that will be requested
        check_local(self._url, settings.allowed_hosts)
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    def _body(self, messages: list[ChatMessage], schema: dict[str, Any]) -> bytes:
        return json.dumps(
            {
                **self._s.extra_body,
                "model": self._s.model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": self._s.max_tokens,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "document", "schema": schema, "strict": True},
                },
            }
        ).encode("utf-8")

    def complete(self, messages: list[ChatMessage], schema: dict[str, Any]) -> VLMReply:
        """Send one request, retrying transient failures.

        Args:
            messages: The chat messages, from build_messages.
            schema: JSON schema the answer must follow.

        Returns:
            The server's answer.

        Raises:
            VLMBadRequest: The server answered HTTP 400, usually a context overflow.
            VLMError: Any other failure, or every attempt failed.
        """
        body = self._body(messages, schema)
        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"

        attempts = self._s.retries + 1
        last = ""
        for attempt in range(attempts):
            if attempt:
                self._sleep(self._s.backoff_s * 2 ** (attempt - 1))
            req = urllib.request.Request(self._url, data=body, headers=headers, method="POST")
            try:
                with self._opener.open(req, timeout=self._s.timeout_s) as resp:
                    raw = resp.read(self._s.max_response_bytes + 1)
            except urllib.error.HTTPError as exc:
                exc.close()  # the error wraps the open response
                last = f"HTTP {exc.code}"
                if exc.code in self._s.retry_statuses:
                    continue
                raise (VLMBadRequest if exc.code == HTTP_BAD_REQUEST else VLMError)(last) from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as exc:
                last = f"{type(exc).__name__}: {exc}"
                continue
            if len(raw) > self._s.max_response_bytes:
                raise VLMError(f"response is larger than vlm.max_response_bytes ({self._s.max_response_bytes})")
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                # A truncated body is often transient, so it shares the retry budget.
                last = f"server returned non JSON: {exc}"
                continue
            return _reply(payload)
        raise VLMError(f"gave up after {attempts} attempts: {last}")
