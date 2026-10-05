"""A local HTTP server that answers like an OpenAI compatible chat endpoint.

Each entry in `responses` is (status, body, delay_s, headers). Request n gets
entry n, and the last entry repeats.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def chat(content: str, model: str = "test-model", finish: str = "stop") -> dict:
    return {"model": model, "choices": [
        {"message": {"role": "assistant", "content": content}, "finish_reason": finish}
    ]}


@contextmanager
def serve(responses):
    requests: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            requests.append({
                "path": self.path,
                "headers": dict(self.headers),
                "body": json.loads(self.rfile.read(n) or b"{}"),
            })
            status, body, delay, headers = responses[min(len(requests), len(responses)) - 1]
            if delay:
                time.sleep(delay)
            payload = (body if isinstance(body, str) else json.dumps(body)).encode("utf-8")
            try:
                self.send_response(status)
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass  # the client timed out and hung up, which some tests intend

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
