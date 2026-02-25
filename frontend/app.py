import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from rag.query import run_query_complete

FRONTEND_DIR = Path(__file__).parent


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
        if self.path != "/api/query":
            self._send(404, json.dumps({"error": "Not found"}))
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            payload = json.loads(raw)
            question = str(payload.get("question", "")).strip()
            if not question:
                self._send(400, json.dumps({"error": "Question is required"}))
                return
            answer, model_name, sources = run_query_complete(question)
            self._send(200, json.dumps({"answer": answer, "model": model_name, "sources": sources}))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self._send(500, json.dumps({"error": str(exc)}))


def main():
    load_dotenv()
    host = os.getenv("FRONTEND_HOST", "127.0.0.1")
    port = int(os.getenv("FRONTEND_PORT", "8000"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Frontend running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
