"use client";

import { useEffect, useRef, useState } from "react";

/**
 * The one place a WebSocket earns its keep.
 *
 * Playback overlay data is static and client-driven, so it travels over
 * cacheable HTTP. A pipeline run is the opposite: the server knows when a stage
 * finishes and the client cannot predict it, so progress is pushed.
 */
export default function JobConsole() {
  const [lines, setLines] = useState([]);
  const [state, setState] = useState("idle");
  const [connected, setConnected] = useState(false);
  const wsRef = useRef(null);
  const boxRef = useRef(null);

  useEffect(() => {
    const url = "ws://" + location.hostname + ":8000/ws/jobs";
    let ws;
    let retry;
    const connect = () => {
      ws = new WebSocket(url);
      wsRef.current = ws;
      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        retry = setTimeout(connect, 3000);
      };
      ws.onmessage = (e) => {
        const m = JSON.parse(e.data);
        if (m.type === "hello" && m.job) {
          setLines(m.job.lines || []);
          setState(m.job.state);
        }
        if (m.type === "start") { setLines([]); setState("running"); }
        if (m.type === "log") setLines((l) => [...l.slice(-400), m.line]);
        if (m.type === "end") setState(m.job.state);
        if (m.type === "error") setLines((l) => [...l, "! " + m.message]);
      };
    };
    connect();
    return () => { clearTimeout(retry); if (ws) ws.close(); };
  }, []);

  useEffect(() => {
    if (boxRef.current) boxRef.current.scrollTop = boxRef.current.scrollHeight;
  }, [lines]);

  const run = (args) => {
    if (wsRef.current && wsRef.current.readyState === 1) {
      wsRef.current.send(JSON.stringify({ type: "run", args }));
    }
  };

  const busy = state === "running" || !connected;

  return (
    <section className="panel wide">
      <header>
        <h3>Pipeline</h3>
        <p>
          <span className={"dot" + (connected ? " on" : "")} />
          {connected ? "socket connected" : "socket offline"} &middot; {state}
        </p>
      </header>
      <div className="panel-body">
        <div className="picker">
          <button
            disabled={busy}
            onClick={() => run(["--video", "Dataset_Video/Intersection_1080p.MP4",
                                "--preview", "20", "--no-render"])}
          >
            20 s preview
          </button>
          <button
            disabled={busy}
            onClick={() => run(["--video", "Dataset_Video/Intersection_1080p.MP4",
                                "--from-tracks", "--no-render"])}
          >
            re-run analytics
          </button>
        </div>
        <pre ref={boxRef} className="console">
          {lines.length ? lines.join("\n") : "no job output yet"}
        </pre>
      </div>
    </section>
  );
}
