"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { ChunkCache, MODE_COLOR, MODES, STATE, idColor, fmtTime } from "@/lib/api";

const TRAIL_FRAMES = 60;

/**
 * Video with a live analysis overlay drawn on a canvas.
 *
 * The overlay is drawn client-side rather than baked into the video, which is
 * what makes it interactive: classes can be toggled, a vehicle can be clicked,
 * and the visualisation can change without re-encoding 800 MB of footage.
 *
 * Sync comes from reading video.currentTime inside a requestAnimationFrame
 * loop. That is deliberate — a timer would drift against the video clock, and
 * the video element is the only authority on where playback actually is.
 */
export default function VideoStage({ meta, filters, onSelect, selectedId }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const cacheRef = useRef(null);
  const trailsRef = useRef(new Map());
  const rafRef = useRef(0);

  const [playing, setPlaying] = useState(false);
  const [t, setT] = useState(0);
  const [rate, setRate] = useState(1);
  const [live, setLive] = useState({});
  const [showTrails, setShowTrails] = useState(true);
  const [showLabels, setShowLabels] = useState(true);

  useEffect(() => {
    cacheRef.current = new ChunkCache(meta.chunk_s, meta.n_chunks, 1);
    cacheRef.current.prefetch(0);
  }, [meta]);

  const draw = useCallback(() => {
    const video = videoRef.current;
    const cv = canvasRef.current;
    if (!video || !cv) return;
    const ctx = cv.getContext("2d");
    const W = meta.width, H = meta.height;
    if (cv.width !== W) { cv.width = W; cv.height = H; }
    ctx.clearRect(0, 0, W, H);

    const time = video.currentTime;
    const frame = Math.round(time * meta.fps);
    const cache = cacheRef.current;
    cache.prefetch(time);
    const rows = cache.frame(frame, meta.fps);

    const counts = {};
    if (rows) {
      const trails = trailsRef.current;
      const visible = [];

      for (const r of rows) {
        const [id, modeIdx, x1, y1, x2, y2, speed, stateIdx] = r;
        const mode = MODES[modeIdx];
        if (!filters.modes[mode]) continue;
        if (filters.minSpeed > 0 && (speed ?? 0) < filters.minSpeed) continue;
        counts[mode] = (counts[mode] || 0) + 1;
        visible.push(r);

        if (STATE[stateIdx] === "moving") {
          let tr = trails.get(id);
          if (!tr) { tr = []; trails.set(id, tr); }
          const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
          const lastPt = tr[tr.length - 1];
          if (!lastPt || lastPt[2] !== frame) {
            tr.push([cx, cy, frame]);
            if (tr.length > TRAIL_FRAMES) tr.shift();
          }
        }
      }

      // drop trails whose vehicle left, and any left behind by a seek
      for (const [id, tr] of trails) {
        const last = tr[tr.length - 1];
        if (!last || Math.abs(frame - last[2]) > TRAIL_FRAMES) trails.delete(id);
      }

      if (showTrails) {
        ctx.lineCap = "round";
        for (const [id, tr] of trails) {
          if (tr.length < 3) continue;
          if (selectedId && id !== selectedId) ctx.globalAlpha = 0.18;
          ctx.strokeStyle = idColor(id);
          for (let i = 1; i < tr.length; i++) {
            const a = i / tr.length;
            ctx.globalAlpha *= 1;
            ctx.lineWidth = Math.max(1, 4 * a);
            ctx.beginPath();
            ctx.globalAlpha = (selectedId && id !== selectedId ? 0.15 : 0.9) * a;
            ctx.moveTo(tr[i - 1][0], tr[i - 1][1]);
            ctx.lineTo(tr[i][0], tr[i][1]);
            ctx.stroke();
          }
          ctx.globalAlpha = 1;
        }
      }

      ctx.font = "600 15px ui-monospace, Menlo, monospace";
      ctx.textBaseline = "alphabetic";
      for (const r of visible) {
        const [id, modeIdx, x1, y1, x2, y2, speed, stateIdx] = r;
        const mode = MODES[modeIdx];
        const sel = selectedId === id;
        const dim = selectedId && !sel;
        ctx.globalAlpha = dim ? 0.25 : 1;

        ctx.strokeStyle = idColor(id);
        ctx.lineWidth = sel ? 4 : 2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        ctx.fillStyle = MODE_COLOR[mode];
        ctx.fillRect(x1, y1, x2 - x1, 4);

        if (showLabels && (x2 - x1 > 26 || sel)) {
          const txt = `${id} ${mode.slice(0, 3)}${speed != null && speed >= 5 ? " " + Math.round(speed) : ""}`;
          const w = ctx.measureText(txt).width + 8;
          ctx.fillStyle = idColor(id);
          ctx.fillRect(x1, y1 - 20, w, 19);
          ctx.fillStyle = "#000";
          ctx.fillText(txt, x1 + 4, y1 - 6);
        }
        ctx.globalAlpha = 1;
      }
    }
    setLive(counts);
    setT(time);
  }, [meta, filters, showTrails, showLabels, selectedId]);

  useEffect(() => {
    const loop = () => { draw(); rafRef.current = requestAnimationFrame(loop); };
    rafRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafRef.current);
  }, [draw]);

  const seek = (sec) => {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = Math.max(0, Math.min(meta.duration_s, sec));
    trailsRef.current.clear();     // history before a seek is not history now
  };

  const onCanvasClick = (e) => {
    const cv = canvasRef.current;
    const rect = cv.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * meta.width;
    const y = ((e.clientY - rect.top) / rect.height) * meta.height;
    const frame = Math.round((videoRef.current?.currentTime || 0) * meta.fps);
    const rows = cacheRef.current?.frame(frame, meta.fps) || [];
    let hit = null;
    for (const r of rows) {
      const [id, , x1, y1, x2, y2] = r;
      if (x >= x1 - 4 && x <= x2 + 4 && y >= y1 - 4 && y <= y2 + 4) hit = id;
    }
    onSelect(hit === selectedId ? null : hit);
  };

  return (
    <div className="stage">
      <div className="video-wrap" style={{ aspectRatio: `${meta.width}/${meta.height}` }}>
        <video
          ref={videoRef}
          src="/api/video"
          preload="auto"
          playsInline
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onSeeked={() => trailsRef.current.clear()}
        />
        <canvas ref={canvasRef} onClick={onCanvasClick} />
        <div className="hud">
          <div className="hud-time">{fmtTime(t)} <span>/ {fmtTime(meta.duration_s)}</span></div>
          {MODES.filter((m) => live[m]).map((m) => (
            <div key={m} className="hud-row">
              <i style={{ background: MODE_COLOR[m] }} />
              <span>{m}</span><b>{live[m]}</b>
            </div>
          ))}
          {!Object.keys(live).length && <div className="hud-row muted">no objects in frame</div>}
        </div>
      </div>

      <div className="controls">
        <button onClick={() => (playing ? videoRef.current.pause() : videoRef.current.play())}>
          {playing ? "Pause" : "Play"}
        </button>
        <button onClick={() => seek(t - 5)}>-5s</button>
        <button onClick={() => seek(t + 5)}>+5s</button>
        <input
          type="range" min={0} max={meta.duration_s} step={0.1} value={t}
          onChange={(e) => seek(parseFloat(e.target.value))}
          className="scrub"
        />
        <select value={rate} onChange={(e) => {
          const r = parseFloat(e.target.value);
          setRate(r); videoRef.current.playbackRate = r;
        }}>
          {[0.25, 0.5, 1, 2, 4].map((r) => <option key={r} value={r}>{r}x</option>)}
        </select>
        <label><input type="checkbox" checked={showTrails}
          onChange={(e) => setShowTrails(e.target.checked)} /> trails</label>
        <label><input type="checkbox" checked={showLabels}
          onChange={(e) => setShowLabels(e.target.checked)} /> labels</label>
      </div>
    </div>
  );
}
