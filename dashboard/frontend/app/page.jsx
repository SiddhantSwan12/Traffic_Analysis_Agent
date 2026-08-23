"use client";

import dynamic from "next/dynamic";
import { useEffect, useState } from "react";
import VideoStage from "@/components/VideoStage";
import Insights from "@/components/Insights";
import JobConsole from "@/components/JobConsole";
import TimeSpace from "@/components/TimeSpace";
import { getJSON, MODES, MODE_COLOR } from "@/lib/api";

// Leaflet reaches for `window` at import time, so the map never server-renders.
const MapPanel = dynamic(() => import("@/components/MapPanel"), {
  ssr: false,
  loading: () => <div className="mapload">loading map…</div>,
});

const TABS = ["Video", "Map", "Insights", "Flow", "Pipeline"];

export default function Page() {
  const [meta, setMeta] = useState(null);
  const [insights, setInsights] = useState(null);
  const [flow, setFlow] = useState(null);
  const [seekTo, setSeekTo] = useState(null);
  const [err, setErr] = useState(null);
  const [tab, setTab] = useState("Video");
  const [selectedId, setSelectedId] = useState(null);
  const [selectedTrack, setSelectedTrack] = useState(null);
  const [filters, setFilters] = useState({
    modes: Object.fromEntries(MODES.map((m) => [m, true])),
    minSpeed: 0,
  });

  useEffect(() => {
    (async () => {
      try {
        for (let i = 0; i < 90; i++) {
          const h = await getJSON("/api/health");
          if (h.error) throw new Error(h.error);
          if (h.ready) break;
          await new Promise((r) => setTimeout(r, 1000));
        }
        setMeta(await getJSON("/api/meta"));
        setInsights(await getJSON("/api/insights"));
        getJSON("/api/flow").then(setFlow).catch(() => setFlow(null));
      } catch (e) {
        setErr(String(e.message || e));
      }
    })();
  }, []);

  useEffect(() => {
    if (!selectedId) { setSelectedTrack(null); return; }
    getJSON(`/api/track/${selectedId}`).then(setSelectedTrack).catch(() => setSelectedTrack(null));
  }, [selectedId]);

  if (err) {
    return (
      <main className="boot">
        <h1>Backend unavailable</h1>
        <p>{err}</p>
        <code>cd dashboard/backend &amp;&amp; python main.py</code>
      </main>
    );
  }
  if (!meta || !insights) {
    return <main className="boot"><h1>Loading analysis…</h1><p>first load parses the track table</p></main>;
  }

  const t = insights.totals || {};
  const geo = insights.georeference || {};

  return (
    <main className="app">
      <header className="top">
        <div className="brand">
          <h1>Traffic Insight</h1>
          <span>{meta.name}</span>
        </div>
        <div className="stats">
          <Stat k="road users" v={t.tracks_moving} />
          <Stat k="journeys" v={t.movements_assigned} />
          <Stat k="approaches" v={(insights.approaches || []).length} />
          <Stat k="duration" v={`${Math.floor(t.duration_s / 60)}m ${Math.round(t.duration_s % 60)}s`} />
          <Stat k="calibration" v={insights.network ? "homography" : "—"} />
        </div>
        <nav>
          {TABS.map((x) => (
            <button key={x} className={x === tab ? "on" : ""} onClick={() => setTab(x)}>{x}</button>
          ))}
        </nav>
      </header>

      <div className="body">
        <aside className="side">
          <h4>Filter modes</h4>
          {MODES.map((m) => (
            <label key={m} className="mode">
              <input
                type="checkbox"
                checked={filters.modes[m]}
                onChange={(e) =>
                  setFilters({ ...filters, modes: { ...filters.modes, [m]: e.target.checked } })
                }
              />
              <i style={{ background: MODE_COLOR[m] }} />
              {m}
            </label>
          ))}

          <h4>Min speed</h4>
          <input
            type="range" min={0} max={50} step={1} value={filters.minSpeed}
            onChange={(e) => setFilters({ ...filters, minSpeed: +e.target.value })}
          />
          <div className="muted">{filters.minSpeed} km/h</div>

          {selectedTrack && (
            <div className="detail">
              <h4>Vehicle #{selectedTrack.display_id}</h4>
              <Row k="mode" v={`${selectedTrack.mode} (${(selectedTrack.mode_confidence * 100).toFixed(0)}%)`} />
              <Row k="source" v={selectedTrack.classification_source} />
              <Row k="length" v={selectedTrack.estimated_length_m ? `${selectedTrack.estimated_length_m} m` : "—"} />
              <Row k="median" v={selectedTrack.median_speed_kmh != null ? `${selectedTrack.median_speed_kmh} km/h` : "—"} />
              <Row k="max" v={selectedTrack.max_speed_kmh != null ? `${selectedTrack.max_speed_kmh} km/h` : "—"} />
              <Row k="movement" v={selectedTrack.origin ? `${selectedTrack.origin} → ${selectedTrack.destination}` : "partial"} />
              <Row k="turn" v={selectedTrack.turn || "—"} />
              <button className="clear" onClick={() => setSelectedId(null)}>clear</button>
            </div>
          )}

          <div className="geo">
            <h4>Georeference</h4>
            <Row k="origin" v={`${geo.origin_lat}, ${geo.origin_lon}`} />
            <Row k="+Y bearing" v={`${geo.local_plus_y_compass_bearing_deg}°`} />
            <Row k="±100 m" v={`${geo.position_uncertainty_at_100m_m} m`} />
          </div>
        </aside>

        <div className="content">
          <div style={{ display: tab === "Video" ? "block" : "none" }}>
            <VideoStage
              meta={meta} filters={filters} seekTo={seekTo}
              onSelect={setSelectedId} selectedId={selectedId}
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
                  ? `${insights.od[0].origin}->${insights.od[0].destination}` : null}
                onSeek={(t, id) => { setSeekTo({ t, at: Date.now() }); setSelectedId(id); setTab("Video"); }}
              />
            ) : <div className="mapload">loading flow analysis…</div>
          )}
          {tab === "Pipeline" && <JobConsole />}
        </div>
      </div>
    </main>
  );
}

function Stat({ k, v }) {
  return <div className="stat"><b>{v ?? "—"}</b><span>{k}</span></div>;
}
function Row({ k, v }) {
  return <div className="row"><span>{k}</span><b>{v}</b></div>;
}
