"""FastAPI backend for the traffic-insight dashboard.

Two transports, each where it belongs.

HTTP serves the video and all analysis data. The video goes out with byte-range
support so the browser can seek natively; overlay data goes out in fixed time
chunks so the browser can cache it, which makes scrubbing backwards free.

WebSocket serves only pipeline job progress -- the one thing here that is
genuinely server-push, because the client cannot know when a stage finishes.
"""
from __future__ import annotations

import asyncio
import json
import mimetypes
import re
import subprocess
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from store import Store, ROOT, IDX_MODE          # noqa: E402

app = FastAPI(title="FlytBase Traffic Insight API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

STORE = Store()


@app.on_event("startup")
def _startup():
    t0 = time.time()
    try:
        STORE.load()
        print(f"[store] ready in {time.time()-t0:.1f}s  "
              f"{STORE.n_chunks()} overlay chunks, "
              f"{len(STORE.net.gates)} approaches")
    except Exception as e:                        # keep the API up to report why
        STORE.error = f"{type(e).__name__}: {e}"
        print(f"[store] FAILED: {STORE.error}")


def _guard():
    if STORE.error:
        raise HTTPException(503, f"pipeline outputs not loaded - {STORE.error}")
    if not STORE.ready:
        raise HTTPException(503, "still loading")


# ----------------------------------------------------------------------- meta
@app.get("/api/health")
def health():
    return {"ready": STORE.ready, "error": STORE.error}


@app.get("/api/meta")
def meta():
    _guard()
    m = STORE.meta()
    return {**m.__dict__, "modes": IDX_MODE,
            "n_chunks": STORE.n_chunks(),
            "totals": STORE.insights.get("totals", {})}


# ---------------------------------------------------------------------- video
RANGE_RE = re.compile(r"bytes=(\d+)-(\d*)")


@app.get("/api/video")
def video(request: Request):
    """Byte-range video streaming.

    Implemented explicitly rather than with FileResponse because seeking in a
    large mp4 depends on 206 Partial Content; without it the browser downloads
    768 MB before it will let anyone scrub.
    """
    _guard()
    path: Path = STORE.video_path
    if not path.exists():
        raise HTTPException(404, "video not found")
    size = path.stat().st_size
    ctype = mimetypes.guess_type(str(path))[0] or "video/mp4"

    rng = request.headers.get("range")
    if not rng:
        return StreamingResponse(
            _read(path, 0, size - 1), media_type=ctype,
            headers={"Content-Length": str(size), "Accept-Ranges": "bytes"})

    m = RANGE_RE.match(rng)
    if not m:
        raise HTTPException(416, "bad range")
    start = int(m.group(1))
    end = int(m.group(2)) if m.group(2) else min(start + 4 * 1024 * 1024 - 1, size - 1)
    end = min(end, size - 1)
    if start > end:
        raise HTTPException(416, "range out of bounds")
    return StreamingResponse(
        _read(path, start, end), status_code=206, media_type=ctype,
        headers={"Content-Range": f"bytes {start}-{end}/{size}",
                 "Accept-Ranges": "bytes",
                 "Content-Length": str(end - start + 1)})


def _read(path: Path, start: int, end: int, block: int = 512 * 1024):
    with open(path, "rb") as f:
        f.seek(start)
        left = end - start + 1
        while left > 0:
            data = f.read(min(block, left))
            if not data:
                break
            left -= len(data)
            yield data


# -------------------------------------------------------------------- overlay
@app.get("/api/chunk/{idx}")
def chunk(idx: int):
    """Overlay boxes for one time chunk.

    Immutable once computed, so it is marked cacheable for a year; the browser
    then re-renders a rewatched segment with no network at all.
    """
    _guard()
    return Response(
        content=json.dumps(STORE.chunk(idx), separators=(",", ":")),
        media_type="application/json",
        headers={"Cache-Control": "public, max-age=31536000, immutable"})


@app.get("/api/track/{display_id}")
def track(display_id: int):
    _guard()
    d = STORE.track_detail(display_id)
    if not d:
        raise HTTPException(404, "no such track")
    return d


# ------------------------------------------------------------------- insights
@app.get("/api/insights")
def insights():
    _guard()
    return JSONResponse(STORE.insights)


@app.get("/api/insights/{key}")
def insight(key: str):
    _guard()
    if key not in STORE.insights:
        raise HTTPException(404, f"unknown insight '{key}'")
    return JSONResponse({key: STORE.insights[key]})


@app.get("/api/map/{kind}")
def geojson(kind: str):
    _guard()
    if kind not in ("approaches", "desire_lines", "queues", "trajectories"):
        raise HTTPException(404, "unknown layer")
    return JSONResponse(STORE.geojson(kind))


# ------------------------------------------------------------------ websocket
class Jobs:
    """Pipeline runs, with progress pushed to every connected client.

    This is the only place a socket earns its keep: a stage boundary is known
    to the server and cannot be predicted by the client.
    """

    def __init__(self):
        self.clients: set[WebSocket] = set()
        self.current: dict | None = None

    async def join(self, ws: WebSocket):
        await ws.accept()
        self.clients.add(ws)
        await ws.send_json({"type": "hello", "job": self.current})

    def leave(self, ws: WebSocket):
        self.clients.discard(ws)

    async def broadcast(self, msg: dict):
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.leave(ws)

    # Jobs are sandboxed to their own directory. A dashboard button must never
    # be able to overwrite the analysis the dashboard is serving: a 20-second
    # preview run writes the same filenames as a full run, and clicking it
    # silently replaced a 6.5-minute result with a 20-second one.
    JOB_OUTDIR = "output/_jobs"

    def _sandbox(self, args: list[str]) -> list[str]:
        out = [a for a in args]
        if "--outdir" in out:
            i = out.index("--outdir")
            if i + 1 < len(out):
                if not out[i + 1].startswith(self.JOB_OUTDIR):
                    out[i + 1] = self.JOB_OUTDIR
                return out
        return out + ["--outdir", self.JOB_OUTDIR]

    async def run(self, args: list[str]):
        args = self._sandbox(args)
        if self.current and self.current.get("state") == "running":
            await self.broadcast({"type": "error",
                                  "message": "a job is already running"})
            return
        self.current = {"state": "running", "args": args,
                        "started": time.time(), "lines": []}
        await self.broadcast({"type": "start", "job": self.current})
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-u", "run.py", *args, cwd=str(ROOT),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        assert proc.stdout is not None
        while True:
            raw = await proc.stdout.readline()
            if not raw:
                break
            line = raw.decode(errors="ignore").rstrip()
            if not line or "deprecated" in line:
                continue
            self.current["lines"].append(line)
            await self.broadcast({"type": "log", "line": line})
        rc = await proc.wait()
        self.current["state"] = "done" if rc == 0 else "failed"
        self.current["returncode"] = rc
        await self.broadcast({"type": "end", "job": self.current})


JOBS = Jobs()


@app.websocket("/ws/jobs")
async def ws_jobs(ws: WebSocket):
    await JOBS.join(ws)
    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("type") == "run":
                args = msg.get("args") or ["--video", "Dataset_Video/Intersection_1080p.MP4",
                                           "--preview", "20", "--no-render"]
                asyncio.create_task(JOBS.run([str(a) for a in args]))
            elif msg.get("type") == "ping":
                await ws.send_json({"type": "pong"})
    except WebSocketDisconnect:
        JOBS.leave(ws)
    except Exception:
        JOBS.leave(ws)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
