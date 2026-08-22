"use client";

import { useEffect, useRef, useState } from "react";
import { getJSON } from "@/lib/api";

const TURN_COLOR = { through: "#4aa3ff", left: "#c778ff", right: "#ffa04a", "u-turn": "#ff5a5a" };

/**
 * Map-native view of the L4 outputs.
 *
 * Leaflet is loaded on the client only, because it touches `window` at import
 * time. Tiles come from OSM; if they fail -- offline, or blocked -- the vector
 * layers still render over the dark background, so the geometry stays usable
 * rather than the panel going blank.
 */
export default function MapPanel({ selectedTrack }) {
  const elRef = useRef(null);
  const mapRef = useRef(null);
  const layersRef = useRef({});
  const selRef = useRef(null);
  const [show, setShow] = useState({
    desire_lines: true, approaches: true, queues: true, trajectories: false,
  });
  const [tilesOk, setTilesOk] = useState(true);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const L = (await import("leaflet")).default;
      await import("leaflet/dist/leaflet.css");
      if (cancelled || mapRef.current || !elRef.current) return;

      const map = L.map(elRef.current, { zoomControl: true, attributionControl: false });
      mapRef.current = map;
      const tiles = L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 20, crossOrigin: true,
      });
      tiles.on("tileerror", () => setTilesOk(false));
      tiles.addTo(map);

      const [desire, appr, queues, traj] = await Promise.all([
        getJSON("/api/map/desire_lines"),
        getJSON("/api/map/approaches"),
        getJSON("/api/map/queues"),
        getJSON("/api/map/trajectories"),
      ]);

      layersRef.current.trajectories = L.geoJSON(traj, {
        style: () => ({ color: "#5a6577", weight: 1, opacity: 0.35 }),
      });
      layersRef.current.desire_lines = L.geoJSON(desire, {
        style: (f) => ({
          color: TURN_COLOR[f.properties.turn] || "#4aa3ff",
          weight: Math.max(2, Math.sqrt(f.properties.count) * 1.6),
          opacity: 0.85,
        }),
        onEachFeature: (f, l) => l.bindPopup(
          "<b>" + f.properties.origin + " to " + f.properties.destination + "</b><br/>" +
          f.properties.turn + " &middot; " + f.properties.count +
          " vehicles (" + f.properties.share_pct + "%)"),
      });
      layersRef.current.queues = L.geoJSON(queues, {
        style: () => ({ color: "#ff5a5a", weight: 9, opacity: 0.55, lineCap: "butt" }),
        onEachFeature: (f, l) => l.bindPopup(
          "<b>" + f.properties.approach + " queue</b><br/>peak " +
          f.properties.peak_queue_veh + " veh &middot; " + f.properties.peak_queue_m +
          " m<br/>at t=" + f.properties.at_t_s + "s"),
      });
      layersRef.current.approaches = L.geoJSON(appr, {
        pointToLayer: (f, latlng) => L.circleMarker(latlng, {
          radius: 9, color: "#fff", weight: 2, fillColor: "#1e88e5", fillOpacity: 0.9,
        }).bindTooltip(f.properties.name, {
          permanent: true, direction: "top", className: "gate-label",
        }),
        onEachFeature: (f, l) => l.bindPopup(
          "<b>Approach " + f.properties.name + "</b><br/>entering " +
          f.properties.entering + " &middot; leaving " + f.properties.leaving +
          "<br/>lanes " + f.properties.n_lanes),
      });

      for (const [k, v] of Object.entries(layersRef.current)) {
        if (show[k]) v.addTo(map);
      }
      try {
        map.fitBounds(layersRef.current.desire_lines.getBounds().pad(0.25));
      } catch (e) {
        map.setView([18.5661, 73.7714], 17);
      }
      setReady(true);
    })();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    for (const [k, v] of Object.entries(layersRef.current)) {
      if (show[k] && !map.hasLayer(v)) v.addTo(map);
      if (!show[k] && map.hasLayer(v)) map.removeLayer(v);
    }
  }, [show, ready]);

  // highlight the vehicle selected in the video, on the map
  useEffect(() => {
    (async () => {
      const map = mapRef.current;
      if (!map) return;
      const L = (await import("leaflet")).default;
      if (selRef.current) { map.removeLayer(selRef.current); selRef.current = null; }
      if (!selectedTrack || !selectedTrack.path || !selectedTrack.path.length) return;
      const pts = selectedTrack.path.map((p) => [p[1], p[0]]);
      selRef.current = L.polyline(pts, { color: "#ffd54a", weight: 4, opacity: 0.95 });
      selRef.current.addTo(map);
      map.fitBounds(selRef.current.getBounds().pad(0.4));
    })();
  }, [selectedTrack]);

  return (
    <div className="mappanel">
      <div className="map-layers">
        {Object.keys(show).map((k) => (
          <label key={k}>
            <input
              type="checkbox"
              checked={show[k]}
              onChange={(e) => setShow({ ...show, [k]: e.target.checked })}
            />
            {k.replace("_", " ")}
          </label>
        ))}
        {!tilesOk && <span className="warn">basemap offline &mdash; vectors only</span>}
      </div>
      <div ref={elRef} className="leaflet-host" />
    </div>
  );
}
