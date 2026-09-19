#!/usr/bin/env python3
"""Local viewer for test clips: waveform + word-timestamp overlays per JSON.

Stdlib only (matches the rest of the tools). Serves this repo's clip dirs
plus a small static frontend (index.html / app.js / style.css / vendor/).

Usage:
  python3 visualize.py                 # port 8377
  python3 visualize.py --port 9000

Open http://localhost:8377/ in a browser. No third-party deps; the frontend
vendors wavesurfer.js v7 (see vendor/).
"""
import argparse
import json
import mimetypes
import re
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MEDIA_EXTS = (".mp3", ".webm", ".opus", ".wav")
# candidate files are openasr outputs or the static external reference;
# anything *.json in a clip dir except truth.json
SKIP_JSON = {"truth.json"}


def discover_clips():
    clips = []
    for d in sorted(ROOT.iterdir()):
        if not d.is_dir() or not (d / "truth.json").exists():
            continue
        media = None
        for ext in MEDIA_EXTS:
            m = d / f"{d.name}{ext}"
            if m.exists():
                media = m.name
                break
        candidates = [
            f.name
            for f in sorted(d.glob("*.json"))
            if f.name not in SKIP_JSON
        ]
        clips.append({"name": d.name, "media": media, "candidates": candidates})
    return clips


class Handler(SimpleHTTPRequestHandler):
    def _send(self, status, body, ctype):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")

    def _send_file(self, p: Path):
        size = p.stat().st_size
        ctype = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
        m = self.RANGE_RE.search(self.headers.get("Range", ""))
        if m:
            lo = int(m.group(1)) if m.group(1) else 0
            hi = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
            if lo > hi or lo >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            body = p.read_bytes()[lo : hi + 1]
            self.send_response(206)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {lo}-{hi}/{size}")
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        body = p.read_bytes()
        self.send_response(200)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/clips":
            body = json.dumps(discover_clips()).encode()
            return self._send(200, body, "application/json")
        if path == "/" or path == "":
            return self.redirect("/visualize/index.html")
        # static frontend
        if path.startswith("/visualize/"):
            p = HERE / path[len("/visualize/"):]
            if p.is_file() and p.resolve().is_relative_to(HERE.resolve()):
                return self._send_file(p)
            return self._send(404, b"not found", "text/plain")
        # media + JSON straight out of the clip dirs
        p = (ROOT / path.lstrip("/")).resolve()
        if p.is_file() and p.is_relative_to(ROOT.resolve()):
            return self._send_file(p)
        return self._send(404, b"not found", "text/plain")

    def redirect(self, loc):
        self.send_response(302)
        self.send_header("Location", loc)
        self.end_headers()

    def log_message(self, format, *args):  # noqa: A002 - match signature
        sys.stderr.write("[visualize] " + (format % args) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8377)
    ap.add_argument("--bind", default="127.0.0.1")
    args = ap.parse_args()
    httpd = HTTPServer((args.bind, args.port), Handler)
    print(f"visualize: serving {ROOT} -> http://{args.bind}:{args.port}/  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
