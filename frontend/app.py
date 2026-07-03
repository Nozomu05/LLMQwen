import base64
import json
import os
import re
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests
from dotenv import load_dotenv

FRONTEND_DIR = Path(__file__).parent

_SEARCH_SERVICE_URL = os.getenv("SEARCH_SERVICE_URL", "http://localhost:8010")

# --- vLLM queue monitoring ---
# Keep in sync with VLLM_MAX_QUEUE in serve.sh.
_VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
_MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE", "20"))


def get_vllm_queue_stats() -> dict:
    """
    Read vLLM Prometheus metrics and return queue state.
    Returns {"waiting": int, "running": int, "available": bool, "backend": str}.
    Falls back gracefully when vLLM is unreachable (transformers backend).
    """
    backend = os.getenv("LLM_BACKEND", "transformers").lower()
    if backend != "vllm":
        return {"waiting": 0, "running": 0, "available": True, "backend": "transformers"}

    try:
        url = f"{_VLLM_BASE_URL}/metrics"
        with urllib.request.urlopen(url, timeout=2) as resp:
            text = resp.read().decode("utf-8")

        waiting = 0
        running = 0
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            m = re.match(r"vllm:num_requests_waiting\{[^}]*\}\s+(\S+)", line)
            if m:
                waiting = int(float(m.group(1)))
                continue
            m = re.match(r"vllm:num_requests_running\{[^}]*\}\s+(\S+)", line)
            if m:
                running = int(float(m.group(1)))

        return {
            "waiting": waiting,
            "running": running,
            "available": waiting < _MAX_QUEUE_SIZE,
            "backend": "vllm",
        }
    except Exception:
        # vLLM not reachable — report as unavailable so the frontend shows a warning
        return {"waiting": -1, "running": -1, "available": False, "backend": "vllm_unreachable"}


def extract_text_from_bytes(filename: str, file_bytes: bytes) -> str:
    """Delegate extraction to the search service's /extract endpoint, which owns
    the PDF/DOCX/PPTX/XLSX parsing logic (and its truncation limit)."""
    resp = requests.post(
        f"{_SEARCH_SERVICE_URL}/extract",
        files={"file": (filename, file_bytes)},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["text"]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def _send(self, status, body, content_type="application/json"):
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split('?')[0]

        if path == "/api/queue":
            self._send(200, json.dumps(get_vllm_queue_stats()))
            return

        if path == "/":
            html_path = FRONTEND_DIR / "index.html"
            with open(html_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self._send(200, content, content_type="text/html; charset=utf-8")
        elif path.endswith('.css'):
            css_path = FRONTEND_DIR / path.lstrip('/')
            if css_path.exists():
                with open(css_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self._send(200, content, content_type="text/css")
            else:
                self._send(404, json.dumps({"error": "Not found"}))
        elif path.endswith('.js'):
            js_path = FRONTEND_DIR / path.lstrip('/')
            if js_path.exists():
                with open(js_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self._send(200, content, content_type="application/javascript")
            else:
                self._send(404, json.dumps({"error": "Not found"}))
        else:
            self._send(404, json.dumps({"error": "Not found"}))

    def do_POST(self):
        if self.path == "/api/upload":
            self._handle_upload()
            return
        if self.path != "/api/query":
            self._send(404, json.dumps({"error": "Not found"}))
            return
        try:
            # Reject early when the vLLM queue is at capacity so the user gets
            # an immediate "try again" instead of a browser timeout.
            stats = get_vllm_queue_stats()
            if stats["backend"] == "vllm" and not stats["available"]:
                waiting = stats["waiting"]
                self._send(503, json.dumps({
                    "error": f"Server is at capacity ({waiting} requests queued). Please try again in a few minutes."
                }))
                return

            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            payload = json.loads(raw)
            question = str(payload.get("question", "")).strip()
            if not question:
                self._send(400, json.dumps({"error": "Question is required"}))
                return
            documents = payload.get("documents", [])
            docs_payload = [
                {"filename": d["filename"], "text": d["text"]}
                for d in documents
                if isinstance(d, dict) and d.get("text")
            ]

            try:
                resp = requests.post(
                    f"{_SEARCH_SERVICE_URL}/query",
                    json={"query": question, "documents": docs_payload or None},
                    stream=True,
                    timeout=180,
                )
            except requests.RequestException as exc:
                self._send(502, json.dumps({"error": f"Search service unreachable: {exc}"}))
                return

            if resp.status_code != 200:
                try:
                    detail = resp.json().get("detail", resp.text)
                except Exception:
                    detail = resp.text or f"HTTP {resp.status_code}"
                self._send(resp.status_code, json.dumps({"error": detail}))
                return

            # Proxy the SSE stream from the search service to the browser
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            try:
                for raw in resp.iter_content(chunk_size=None):
                    self.wfile.write(raw)
                    self.wfile.flush()
            except Exception:
                pass

        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            import traceback
            traceback.print_exc()
            try:
                self.wfile.write(f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n".encode("utf-8"))
                self.wfile.flush()
            except Exception:
                pass

    def _handle_upload(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            payload = json.loads(raw)
            filename = str(payload.get("filename", "document"))
            data_url = str(payload.get("data", ""))
            if "," in data_url:
                data_url = data_url.split(",", 1)[1]
            file_bytes = base64.b64decode(data_url)
            text = extract_text_from_bytes(filename, file_bytes)
            self._send(200, json.dumps({"filename": filename, "text": text, "chars": len(text)}))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self._send(500, json.dumps({"error": str(exc)}))


def main():
    load_dotenv()
    host = os.getenv("FRONTEND_HOST", "0.0.0.0")
    port = int(os.getenv("FRONTEND_PORT", "8080"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Frontend running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
