"use client";

import dynamic from "next/dynamic";
import { useEffect, useMemo, useState } from "react";
import VideoStage from "@/components/VideoStage";
import Overview from "@/components/Overview";
import Insights from "@/components/Insights";
import JobConsole from "@/components/JobConsole";
import TimeSpace from "@/components/TimeSpace";
import Reasoning from "@/components/Reasoning";
import { getJSON, MODES, MODE_COLOR } from "@/lib/api";

// Leaflet reaches for `window` at import time, so the map never server-renders.
const MapPanel = dynamic(() => import("@/components/MapPanel"), {
  ssr: false,
  loading: () => <div className="mapload">Preparing network map…</div>,
});

const TABS = [
  { id: "Overview", label: "Overview", icon: "overview", description: "Network performance at a glance" },
  { id: "Video", label: "Live video", icon: "video", description: "Inspect tracked road users" },
  { id: "Map", label: "Network map", icon: "map", description: "Review georeferenced movement" },
  { id: "Insights", label: "Analytics", icon: "analytics", description: "Explore traffic metrics" },
  { id: "Flow", label: "Traffic flow", icon: "flow", description: "Follow trajectories through time" },
  { id: "Reasoning", label: "Findings", icon: "findings", description: "Evidence-backed network findings" },
  { id: "Pipeline", label: "Pipeline", icon: "pipeline", description: "Run and monitor processing" },
];

export default function Page() {
  const [meta, setMeta] = useState(null);
  const [insights, setInsights] = useState(null);
  const [flow, setFlow] = useState(null);
  const [seekTo, setSeekTo] = useState(null);
  const [err, setErr] = useState(null);
  const [tab, setTab] = useState("Overview");
  const [selectedId, setSelectedId] = useState(null);
  const [selectedTrack, setSelectedTrack] = useState(null);
  const [filters, setFilters] = useState({
    modes: Object.fromEntries(MODES.map((mode) => [mode, true])),
    minSpeed: 0,
  });

  useEffect(() => {
    (async () => {
      try {
        for (let i = 0; i < 90; i++) {
          const health = await getJSON("/api/health");
          if (health.error) throw new Error(health.error);
          if (health.ready) break;
          await new Promise((resolve) => setTimeout(resolve, 1000));
        }
        setMeta(await getJSON("/api/meta"));
        setInsights(await getJSON("/api/insights"));
        getJSON("/api/flow").then(setFlow).catch(() => setFlow(null));
      } catch (error) {
        setErr(String(error.message || error));
      }
    })();
  }, []);

  useEffect(() => {
    if (!selectedId) {
      setSelectedTrack(null);
      return;
    }
    getJSON(`/api/track/${selectedId}`)
      .then(setSelectedTrack)
      .catch(() => setSelectedTrack(null));
  }, [selectedId]);

  const activeModes = useMemo(
    () => Object.values(filters.modes).filter(Boolean).length,
    [filters.modes],
  );

  if (err) {
    return (
      <main className="boot boot-error">
        <div className="boot-mark"><Icon name="warning" /></div>
        <p className="eyebrow">Connection issue</p>
        <h1>Analytics backend unavailable</h1>
        <p>{err}</p>
        <code>cd dashboard/backend &amp;&amp; python main.py</code>
      </main>
    );
  }

  if (!meta || !insights) {
    return (
      <main className="boot">
        <div className="loader" aria-label="Loading analysis" />
        <p className="eyebrow">FlytBase traffic intelligence</p>
        <h1>Preparing the network view</h1>
        <p>Parsing trajectories and calculating movement metrics…</p>
      </main>
    );
  }

  const current = TABS.find((item) => item.id === tab) || TABS[0];
  const totals = insights.totals || {};

  const navigate = (nextTab) => {
    setTab(nextTab);
    document.querySelector(".workspace-content")?.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <main className="app-shell">
      <aside className="rail">
        <div className="rail-brand">
          <div className="brand-mark" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>
          <div>
            <b>Traffic Intelligence</b>
            <span>FlytBase Analytics</span>
          </div>
        </div>

        <nav className="rail-nav" aria-label="Dashboard sections">
          <p className="rail-label">Workspace</p>
          {TABS.map((item) => (
            <button
              key={item.id}
              className={item.id === tab ? "active" : ""}
              onClick={() => navigate(item.id)}
              aria-current={item.id === tab ? "page" : undefined}
              title={item.label}
            >
              <Icon name={item.icon} />
              <span>{item.label}</span>
            </button>
          ))}
        </nav>

        <section className="rail-section filters" aria-labelledby="filter-heading">
          <div className="rail-section-head">
            <p id="filter-heading" className="rail-label">Scene filters</p>
            <span>{activeModes}/{MODES.length}</span>
          </div>
          <div className="mode-grid">
            {MODES.map((mode) => (
              <label key={mode} className="mode-filter">
                <input
                  type="checkbox"
                  checked={filters.modes[mode]}
                  onChange={(event) => setFilters({
                    ...filters,
                    modes: { ...filters.modes, [mode]: event.target.checked },
                  })}
                />
                <i style={{ "--mode-color": MODE_COLOR[mode] }} />
                <span>{mode}</span>
                <Icon name="check" />
              </label>
            ))}
          </div>

          <div className="speed-filter">
            <div>
              <span>Minimum speed</span>
              <b>{filters.minSpeed} km/h</b>
            </div>
            <input
              aria-label="Minimum speed"
              type="range"
              min={0}
              max={50}
              step={1}
              value={filters.minSpeed}
              onChange={(event) => setFilters({ ...filters, minSpeed: +event.target.value })}
            />
          </div>
        </section>

        {selectedTrack && (
          <section className="selected-card">
            <div className="selected-card-head">
              <div>
                <span>Selected road user</span>
                <b>Vehicle #{selectedTrack.display_id}</b>
              </div>
              <button className="icon-button" onClick={() => setSelectedId(null)} aria-label="Clear selection">
                <Icon name="close" />
              </button>
            </div>
            <Row k="Mode" v={`${selectedTrack.mode} · ${(selectedTrack.mode_confidence * 100).toFixed(0)}%`} />
            <Row k="Median speed" v={selectedTrack.median_speed_kmh != null ? `${selectedTrack.median_speed_kmh} km/h` : "—"} />
            <Row k="Peak speed" v={selectedTrack.max_speed_kmh != null ? `${selectedTrack.max_speed_kmh} km/h` : "—"} />
            <Row k="Movement" v={selectedTrack.origin ? `${selectedTrack.origin} → ${selectedTrack.destination}` : "Partial"} />
          </section>
        )}

        <div className="rail-footer">
          <span className="status-dot" />
          <div><b>Analysis ready</b><span>{meta.name}</span></div>
        </div>
      </aside>

      <section className="workspace">
        <header className="workspace-top">
          <div>
            <p className="breadcrumb">Traffic intelligence <span>/</span> {current.label}</p>
            <h1>{current.label}</h1>
            <p>{current.description}</p>
          </div>
          <div className="workspace-meta">
            <div className="meta-pill">
              <Icon name="clock" />
              <span>Capture</span>
              <b>{formatDuration(totals.duration_s)}</b>
            </div>
            <div className="meta-pill">
              <Icon name="target" />
              <span>Calibration</span>
              <b>{insights.network ? "Homography" : "Unavailable"}</b>
            </div>
            <div className="live-pill"><span /> System ready</div>
          </div>
        </header>

        <div className="workspace-content">
          {tab === "Overview" && (
            <Overview data={insights} meta={meta} onNavigate={navigate} />
          )}
          <div className={tab === "Video" ? "tab-pane active" : "tab-pane"} aria-hidden={tab !== "Video"}>
            <VideoStage
              meta={meta}
              filters={filters}
              seekTo={seekTo}
              onSelect={setSelectedId}
              selectedId={selectedId}
            />
          </div>
          {tab === "Map" && <MapPanel selectedTrack={selectedTrack} />}
          {tab === "Insights" && <Insights data={insights} />}
          {tab === "Flow" && (
            flow ? (
              <TimeSpace
                corridors={Object.keys(flow.validity || {})}
                validity={flow.validity}
                busiest={(insights.od || []).length
                  ? `${insights.od[0].origin}->${insights.od[0].destination}`
                  : null}
                onSeek={(time, id) => {
                  setSeekTo({ t: time, at: Date.now() });
                  setSelectedId(id);
                  navigate("Video");
                }}
              />
            ) : <div className="mapload">Preparing flow analysis…</div>
          )}
          {tab === "Reasoning" && <Reasoning />}
          {tab === "Pipeline" && <JobConsole />}
        </div>
      </section>
    </main>
  );
}

function formatDuration(seconds = 0) {
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  return `${mins}m ${secs}s`;
}

function Row({ k, v }) {
  return <div className="row"><span>{k}</span><b>{v}</b></div>;
}

function Icon({ name }) {
  const paths = {
    overview: <><rect x="3" y="3" width="7" height="7" rx="2"/><rect x="14" y="3" width="7" height="4" rx="2"/><rect x="14" y="11" width="7" height="10" rx="2"/><rect x="3" y="14" width="7" height="7" rx="2"/></>,
    video: <><rect x="3" y="5" width="14" height="14" rx="3"/><path d="m17 10 4-2v8l-4-2"/></>,
    map: <><path d="m3 6 6-3 6 3 6-3v15l-6 3-6-3-6 3Z"/><path d="M9 3v15M15 6v15"/></>,
    analytics: <><path d="M4 19V9M10 19V5M16 19v-7M22 19H2"/></>,
    flow: <><path d="M4 18V6M4 12h6c4 0 4-6 8-6h2M4 16h6c4 0 4 2 8 2h2"/><path d="m18 3 3 3-3 3m0 6 3 3-3 3"/></>,
    findings: <><path d="M9 18h6M10 22h4"/><path d="M8.5 14.5A7 7 0 1 1 15.5 14.5C14 15.5 14 17 14 18h-4c0-1 0-2.5-1.5-3.5Z"/></>,
    pipeline: <><rect x="3" y="4" width="6" height="6" rx="2"/><rect x="15" y="14" width="6" height="6" rx="2"/><path d="M9 7h4a3 3 0 0 1 3 3v4M15 17h-4a3 3 0 0 1-3-3v-4"/></>,
    check: <path d="m6 12 4 4 8-9"/>,
    close: <path d="m7 7 10 10M17 7 7 17"/>,
    clock: <><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></>,
    target: <><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3"/></>,
    warning: <><path d="M12 3 2.8 20h18.4Z"/><path d="M12 9v4M12 17h.01"/></>,
  };
  return (
    <svg className="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {paths[name] || paths.overview}
    </svg>
  );
}
