"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { getJSON } from "@/lib/api";
import { MODE_COLOR, SPEED_STOPS, CHART } from "@/lib/viz";

/**
 * Time-space diagram: time on x, distance along the corridor on y, one
 * polyline per vehicle.
 *
 * This is the plot traffic engineering reaches for first, because structure no
 * summary statistic shows is directly visible in it. A queue is a flat band. A
 * discharge is a fan. A shockwave is a diagonal front, and the slope of that
 * front IS its propagation speed -- which is how you trace a jam back to where
 * it began.
 *
 * Drawn on canvas rather than with a charting library: several hundred
 * polylines of a few hundred points each is tens of thousands of segments, and
 * an SVG node per segment would make the page unusable.
 */
function speedColor(v) {
  if (v == null || !isFinite(v)) return "#5a6577";
  let c = SPEED_STOPS[0][1];
  for (const [th, col] of SPEED_STOPS) if (v >= th) c = col;
  return c;
}

export default function TimeSpace({ corridors, validity, onSeek, busiest }) {
  const [corridor, setCorridor] = useState(busiest || corridors[0] || "");
  // A whole session squeezed into one canvas is unreadable: 400 s across 1,250
  // px makes every traverse near-vertical and 500 of them merge into noise.
  // pNEUMA plots a window, so this does too.
  const [win, setWin] = useState(90);
  const [t0, setT0] = useState(0);
  const [data, setData] = useState(null);
  const [colorBy, setColorBy] = useState("speed");
  const [loading, setLoading] = useState(false);
  const [hover, setHover] = useState(null);
  const cvRef = useRef(null);
  const boxRef = useRef(null);

  useEffect(() => {
    if (!corridor) return;
    setLoading(true);
    getJSON(`/api/flow/timespace/${encodeURIComponent(corridor)}`)
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, [corridor]);

  const extent = useMemo(() => {
    if (!data?.tracks?.length) return null;
    let a = Infinity, b = -Infinity;
    for (const tr of data.tracks) {
      if (tr.t[0] < a) a = tr.t[0];
      if (tr.t[tr.t.length - 1] > b) b = tr.t[tr.t.length - 1];
    }
    return { a, b };
  }, [data]);

  const bounds = useMemo(() => {
    if (!data || !extent) return null;
    const span = win >= 9999 ? extent.b - extent.a : win;
    const start = Math.max(extent.a, Math.min(t0, extent.b - span));
    return { t0: start, t1: start + span, s0: 0, s1: data.corridor_length_m };
  }, [data, extent, win, t0]);

  // only the trajectories inside the window need drawing
  const visible = useMemo(() => {
    if (!data?.tracks || !bounds) return [];
    return data.tracks.filter(
      (tr) => tr.t[tr.t.length - 1] >= bounds.t0 && tr.t[0] <= bounds.t1);
  }, [data, bounds]);

  useEffect(() => {
    const cv = cvRef.current;
    const box = boxRef.current;
    if (!cv || !box || !data || !bounds) return;

    const W = box.clientWidth;
    const H = 460;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    cv.width = W * dpr;
    cv.height = H * dpr;
    cv.style.width = W + "px";
    cv.style.height = H + "px";
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const PAD = { l: 52, r: 12, t: 10, b: 30 };
    const iw = W - PAD.l - PAD.r;
    const ih = H - PAD.t - PAD.b;
    const X = (t) => PAD.l + ((t - bounds.t0) / (bounds.t1 - bounds.t0)) * iw;
    // distance increases upward, the conventional orientation
    const Y = (s) => PAD.t + ih - ((s - bounds.s0) / (bounds.s1 - bounds.s0)) * ih;

    // grid
    ctx.strokeStyle = "#232a36";
    ctx.fillStyle = "#8a94a6";
    ctx.font = "11px ui-monospace, Menlo, monospace";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 6; i++) {
      const t = bounds.t0 + ((bounds.t1 - bounds.t0) * i) / 6;
      const x = X(t);
      ctx.beginPath(); ctx.moveTo(x, PAD.t); ctx.lineTo(x, PAD.t + ih); ctx.stroke();
      ctx.textAlign = "center";
      ctx.fillText(`${Math.round(t)}s`, x, H - 10);
    }
    for (let i = 0; i <= 5; i++) {
      const s = bounds.s0 + ((bounds.s1 - bounds.s0) * i) / 5;
      const y = Y(s);
      ctx.beginPath(); ctx.moveTo(PAD.l, y); ctx.lineTo(PAD.l + iw, y); ctx.stroke();
      ctx.textAlign = "right";
      ctx.fillText(`${Math.round(s)}m`, PAD.l - 6, y + 4);
    }

    // trajectories
    ctx.lineWidth = 1.15;
    ctx.lineJoin = "round";
    for (const tr of visible) {
      const hl = hover && hover.id === tr.id;
      ctx.globalAlpha = hover ? (hl ? 1 : 0.12) : 0.75;
      if (colorBy === "mode") {
        ctx.strokeStyle = MODE_COLOR[tr.mode] || "#8a94a6";
        ctx.beginPath();
        ctx.moveTo(X(tr.t[0]), Y(tr.s[0]));
        for (let i = 1; i < tr.t.length; i++) ctx.lineTo(X(tr.t[i]), Y(tr.s[i]));
        ctx.stroke();
      } else {
        // colour each segment by instantaneous speed: this is what makes a
        // stopped queue read as a red horizontal band
        for (let i = 1; i < tr.t.length; i++) {
          ctx.strokeStyle = speedColor(tr.v[i]);
          ctx.beginPath();
          ctx.moveTo(X(tr.t[i - 1]), Y(tr.s[i - 1]));
          ctx.lineTo(X(tr.t[i]), Y(tr.s[i]));
          ctx.stroke();
        }
      }
      if (hl) {
        ctx.lineWidth = 2.6;
        ctx.strokeStyle = "#ffd54a";
        ctx.beginPath();
        ctx.moveTo(X(tr.t[0]), Y(tr.s[0]));
        for (let i = 1; i < tr.t.length; i++) ctx.lineTo(X(tr.t[i]), Y(tr.s[i]));
        ctx.stroke();
        ctx.lineWidth = 1.15;
      }
    }
    ctx.globalAlpha = 1;
  }, [data, bounds, colorBy, hover, visible]);

  const pick = (e) => {
    const cv = cvRef.current;
    if (!cv || !data || !bounds) return;
    const r = cv.getBoundingClientRect();
    const PAD = { l: 52, r: 12, t: 10, b: 30 };
    const iw = r.width - PAD.l - PAD.r;
    const ih = 460 - PAD.t - PAD.b;
    const t = bounds.t0 + ((e.clientX - r.left - PAD.l) / iw) * (bounds.t1 - bounds.t0);
    const s = bounds.s0 + ((PAD.t + ih - (e.clientY - r.top)) / ih) * (bounds.s1 - bounds.s0);
    let best = null, bd = Infinity;
    for (const tr of visible) {
      for (let i = 0; i < tr.t.length; i += 2) {
        const dt = (tr.t[i] - t) / (bounds.t1 - bounds.t0);
        const ds = (tr.s[i] - s) / (bounds.s1 - bounds.s0);
        const d = dt * dt + ds * ds;
        if (d < bd) { bd = d; best = { id: tr.id, mode: tr.mode, t: tr.t[i], s: tr.s[i], v: tr.v[i] }; }
      }
    }
    return bd < 0.0009 ? best : null;
  };

  const val = validity?.[corridor];

  return (
    <section className="panel wide">
      <header>
        <h3>Time–space diagram</h3>
        <p>
          one line per vehicle · a flat band is a queue, a diagonal front is a
          shockwave and its slope is the propagation speed
          {val ? ` · flow theory holds at ${val.negative}/${val.locations} locations` : ""}
        </p>
      </header>
      <div className="panel-body">
        <div className="picker">
          {corridors.map((k) => (
            <button key={k} className={k === corridor ? "on" : ""}
                    onClick={() => setCorridor(k)}>{k}</button>
          ))}
          <span className="spacer" />
          <button className={colorBy === "speed" ? "on" : ""}
                  onClick={() => setColorBy("speed")}>by speed</button>
          <button className={colorBy === "mode" ? "on" : ""}
                  onClick={() => setColorBy("mode")}>by mode</button>
        </div>

        <div className="ts-window">
          <span>window</span>
          {[30, 60, 90, 150, 9999].map((w) => (
            <button key={w} className={w === win ? "on" : ""} onClick={() => setWin(w)}>
              {w === 9999 ? "all" : `${w}s`}
            </button>
          ))}
          <input
            type="range" min={0} step={1}
            max={Math.max(0, (extent ? extent.b - extent.a : 0) - Math.min(win, 1e4))}
            value={t0} onChange={(e) => setT0(+e.target.value)}
            disabled={win >= 9999}
          />
          <b>{bounds ? `${bounds.t0.toFixed(0)}–${bounds.t1.toFixed(0)}s` : "—"}</b>
        </div>

        <div ref={boxRef} className="ts-wrap">
          {loading && <div className="ts-load">loading trajectories…</div>}
          <canvas
            ref={cvRef}
            onMouseMove={(e) => setHover(pick(e))}
            onMouseLeave={() => setHover(null)}
            onClick={(e) => { const h = pick(e); if (h && onSeek) onSeek(h.t, h.id); }}
          />
          {hover && (
            <div className="ts-tip">
              <b>#{hover.id}</b> {hover.mode} · {hover.s.toFixed(0)} m ·{" "}
              {hover.v != null ? `${hover.v.toFixed(0)} km/h` : "—"} · t={hover.t.toFixed(1)}s
              <em>click to jump the video here</em>
            </div>
          )}
        </div>

        {colorBy === "speed" && (
          <div className="ts-legend">
            {SPEED_STOPS.map(([th, c], i) => (
              <span key={th}>
                <i style={{ background: c }} />
                {i === SPEED_STOPS.length - 1 ? `${th}+` : `${th}–${SPEED_STOPS[i + 1][0]}`} km/h
              </span>
            ))}
          </div>
        )}
        {data && (
          <div className="muted" style={{ marginTop: 8 }}>
            showing {visible.length} of {data.n_tracks} vehicles over{" "}
            {data.corridor_length_m} m
          </div>
        )}
      </div>
    </section>
  );
}
