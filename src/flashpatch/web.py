from __future__ import annotations

import json
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .product import repair_video, scan_video, verify_video

PAGE = b"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>FlashPatch Visual Safety Demo</title><style>
body{font:16px system-ui;max-width:760px;margin:3rem auto;padding:0 1rem}button{padding:.7rem 1rem}
pre{white-space:pre-wrap;background:#f4f4f4;padding:1rem}video{max-width:100%}
</style></head><body><h1>FlashPatch</h1>
<p>WCAG 2.2 video hazard scan, minimal local repair, and independent verification.</p>
<label>Upload video <input id="video" type="file" accept="video/mp4"></label>
<button id="run">Scan, repair, verify</button><pre id="result">Ready.</pre>
<a id="download" hidden>Download repaired video</a>
<script>document.querySelector('#run').onclick=async()=>{const f=document.querySelector('#video').files[0];
if(!f)return;const r=await fetch('/api/repair',{method:'POST',headers:{'Content-Type':'video/mp4','X-FlashPatch-Filename':f.name},body:f});
const p=await r.json();document.querySelector('#result').textContent=JSON.stringify(p,null,2);
const a=document.querySelector('#download');if(p.download_url){a.href=p.download_url;a.hidden=false;}};</script></body></html>"""


def _safe_name(value: str) -> str:
    name = Path(value).name
    return re.sub(r"[^A-Za-z0-9._-]", "_", name) or "upload.mp4"


def create_server(host: str, port: int, *, workspace: str | Path) -> ThreadingHTTPServer:
    root = Path(workspace).resolve()
    root.mkdir(parents=True, exist_ok=True)

    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
            body = json.dumps(payload, sort_keys=True).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            path = unquote(urlparse(self.path).path)
            if path == "/":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(PAGE)))
                self.end_headers()
                self.wfile.write(PAGE)
                return
            if path.startswith("/downloads/"):
                target = (root / Path(path).name).resolve()
                if target.parent == root and target.is_file():
                    body = target.read_bytes()
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "video/mp4")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Content-Disposition", f'attachment; filename="{target.name}"')
                    self.end_headers()
                    self.wfile.write(body)
                    return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/api/repair":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 512 * 1024 * 1024:
                self._json(HTTPStatus.BAD_REQUEST, {"error": "video body is required"})
                return
            name = _safe_name(self.headers.get("X-FlashPatch-Filename", "upload.mp4"))
            source = root / name
            repaired = root / f"{source.stem}-repaired.mp4"
            receipt = root / f"{source.stem}-receipt.json"
            source.write_bytes(self.rfile.read(length))
            try:
                payload = {
                    "scan": scan_video(source),
                    "repair": repair_video(source, repaired, receipt=receipt),
                    "verification": verify_video(repaired),
                    "download_url": f"/downloads/{repaired.name}",
                }
            except Exception as error:
                self._json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(error)})
                return
            self._json(HTTPStatus.OK, payload)

        def log_message(self, format: str, *args: object) -> None:
            return

    return ThreadingHTTPServer((host, port), Handler)


def serve(host: str, port: int, *, workspace: str | Path) -> None:
    server = create_server(host, port, workspace=workspace)
    try:
        server.serve_forever()
    finally:
        server.server_close()