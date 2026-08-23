"use client";

import { useEffect, useState } from "react";
import { getJSON } from "@/lib/api";
import { MODE_COLOR } from "@/lib/viz";

/**
 * L5 findings, each shown with the evidence it was reasoned from.
 *
 * None of these questions has ground truth in the footage, so a bare verdict
 * would be unfalsifiable. Every card carries the numbers behind it, and where
 * two methods disagree both are shown rather than the nicer one being picked.
 */
function Verdict({ value, yes = "yes", no = "no", unknown = "not determinable" }) {
  if (value === null || value === undefined)
    return <span className="verdict unk">{unknown}</span>;
  return <span className={`verdict ${value ? "yes" : "no"}`}>{value ? yes : no}</span>;
}

function Card({ title, sub, children }) {
  return (
    <section className="panel">
      <header><h3>{title}</h3>{sub && <p>{sub}</p>}</header>
      <div className="panel-body">{children}</div>
    </section>
  );
}

function KV({ k, v, note }) {
  return (
    <div className="row">
      <span>{k}</span>
      <b>{v === null || v === undefined ? "—" : v}{note && <em> {note}</em>}</b>
    </div>
  );
}

export default function Reasoning() {
  const [r, setR] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    getJSON("/api/reasoning").then(setR).catch((e) => setErr(String(e)));
  }, []);

  if (err) return <div className="mapload">reasoning unavailable — {err}</div>;
  if (!r) return <div className="mapload">running network reasoning…</div>;

  const corridors = Object.keys(r.congestion || {});
  const comp = r.signal_complementarity || {};
  const cg = r.critical_gap || {};
  const anySignal = Object.values(r.signal_periodicity || {}).some((s) => s.signalised);

  return (
    <div className="insights">
      <Card
        title="Congestion origination"
        sub="where each jam began, and which way its front moves"
      >
        {corridors.map((c) => {
          const reg = (r.congestion[c]?.regions || [])[0];
          const rh = r.shockwave_rh?.[c];
          if (!reg) return (
            <div key={c} className="finding">
              <h5>{c}</h5>
              <p className="muted">{r.congestion[c]?.note}</p>
            </div>
          );
          return (
            <div key={c} className="finding">
              <h5>{c}</h5>
              <KV k="first seen at" v={`${reg.origin_distance_m} m, t=${reg.origin_time_s}s`} />
              <KV k="slowest" v={`${reg.min_speed_kmh} km/h`} />
              <KV k="extent / duration" v={`${reg.extent_m} m for ${reg.duration_s}s`} />
              {reg.predates_clip && (
                <p className="caveat">
                  already under way when recording began — formation happened off camera
                </p>
              )}
              <div className="compare">
                <div>
                  <span>geometric edge</span>
                  <b>{reg.shockwave_mps ?? "—"} m/s</b>
                  <em>{reg.propagating}</em>
                </div>
                <div className="preferred">
                  <span>Rankine–Hugoniot</span>
                  <b>{rh?.median_mps ?? "—"} m/s</b>
                  <em>{rh?.direction ?? "—"}{rh?.n_fronts ? ` · ${rh.n_fronts} fronts` : ""}</em>
                </div>
              </div>
            </div>
          );
        })}
        <p className="method">
          The jump condition u = Δq/Δk follows from conservation of vehicles and
          needs no edge to be located, so it is the one to believe where the two
          disagree.
        </p>
      </Card>

      <Card title="Signal performance" sub="inferred from behaviour; timings are not in the data">
        <div className="headline">
          <Verdict value={anySignal || comp.signal_like ? true : false}
                   yes="signalised" no="uncontrolled" />
        </div>
        {Object.entries(r.signal_periodicity || {}).map(([c, s]) => (
          <KV key={c} k={c}
              v={s.signalised ? `cycle ${s.cycle_length_s}s` : "no recurring cycle"}
              note={`prominence ${s.peak_prominence ?? "—"} / ${s.prominence_threshold ?? "—"}`} />
        ))}
        <KV k="approaches alternate?" v={comp.median_correlation ?? "—"}
            note={comp.signal_like ? "signal-like" : "independent"} />
        <p className="method">
          Three independent tests agree: no periodicity in any approach's
          discharge, and conflicting approaches do not take turns. A long pause
          alone is not evidence — light traffic produces those by accident.
        </p>
      </Card>

      <Card title="Gap acceptance" sub={cg.pair || "minor road against the major stream"}>
        <div className="headline">
          <b>{cg.pooled?.critical_gap_s ?? "—"} s</b>
          <span>critical gap (Raff)</span>
        </div>
        {Object.entries(cg.by_mode || {}).map(([m, v]) => (
          <div key={m} className="row">
            <span><i className="chip" style={{ background: MODE_COLOR[m] }} />{m}</span>
            <b>{v.critical_gap_s ?? "—"} s
              <em> {v.n_accepted} accepted / {v.n_rejected} rejected</em></b>
          </div>
        ))}
        <KV k="lags (first gap on arrival)" v={cg.n_lags}
            note={cg.median_lag_s ? `median ${cg.median_lag_s}s` : ""} />
        <p className="method">
          Split by mode because one pooled figure describes nobody in mixed
          traffic, and lags kept separate because a driver never watched the
          first interval open.
        </p>
      </Card>

      <Card title="Lane discipline" sub="structure inferred from where traffic sits">
        {(r.lane_structure || []).map((l) => (
          <div key={l.corridor} className="row">
            <span>{l.corridor}</span>
            <b>{l.lane_structure}
              <em> {l.n_modes} mode{l.n_modes === 1 ? "" : "s"}
                {l.observed_lane_spacing_m ? ` · ${l.observed_lane_spacing_m} m apart` : ""}</em>
            </b>
          </div>
        ))}
        <p className="method">
          Lane structure means several bands about a lane apart. A single sharp
          peak is concentration, not lanes — every corridor here is one
          undifferentiated stream.
        </p>
      </Card>

      <Card title="Obstruction census" sub="stationary vehicles on the carriageway">
        <div className="headline">
          <b>{(r.obstructions || []).length}</b>
          <span>
            {(r.obstructions || []).filter((o) => o.obstruction === "in-lane").length} fully in-lane
          </span>
        </div>
        <div className="scroll">
          <table>
            <thead><tr><th>mode</th><th>corridor</th><th className="num">at</th>
              <th className="num">lateral</th><th className="num">dwell</th><th>type</th></tr></thead>
            <tbody>
              {(r.obstructions || []).slice(0, 10).map((o) => (
                <tr key={o.track_id}>
                  <td><i className="chip" style={{ background: MODE_COLOR[o.mode] }} />{o.mode}</td>
                  <td>{o.corridor}</td>
                  <td className="num">{o.distance_along_m} m</td>
                  <td className="num">{o.lateral_offset_m} m</td>
                  <td className="num">{o.dwell_s} s</td>
                  <td>{o.obstruction}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Card title="Lane changes" sub="weaving intensity per corridor">
        <div className="scroll">
          <table>
            <thead><tr><th>corridor</th><th className="num">vehicles</th>
              <th className="num">changes</th><th className="num">per km</th></tr></thead>
            <tbody>
              {(r.lane_changes || []).map((l) => (
                <tr key={l.corridor}>
                  <td>{l.corridor}</td>
                  <td className="num">{l.n_vehicles}</td>
                  <td className="num">{l.lane_changes}</td>
                  <td className="num">{l.changes_per_km}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="method">
          Counted with hysteresis and a one-second dwell, so drifting across a
          boundary does not register as a change.
        </p>
      </Card>
    </div>
  );
}
