"use client";

import { useMemo, useState } from "react";
import {
  ResponsiveContainer, Line, BarChart, Bar, ScatterChart, Scatter,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, Cell, AreaChart, Area,
  ComposedChart, ReferenceLine,
} from "recharts";
import {
  SERIES, MODE_COLOR, MODES, TURN_COLOR, seqColor, STATUS,
  AXIS, CHART, TOOLTIP_STYLE, BAR_RADIUS, BAR_RADIUS_H, fmt,
} from "@/lib/viz";

function Panel({ title, sub, children, wide }) {
  return (
    <section className={`panel${wide ? " wide" : ""}`}>
      <header><h3>{title}</h3>{sub && <p>{sub}</p>}</header>
      <div className="panel-body">{children}</div>
    </section>
  );
}

/** The number IS the chart. A one-bar bar chart would say less. */
function Tile({ value, unit, label, note, tone }) {
  return (
    <div className="tile">
      <b style={tone ? { color: tone } : undefined}>{value}{unit && <i>{unit}</i>}</b>
      <span>{label}</span>
      {note && <em>{note}</em>}
    </div>
  );
}

function Tip({ active, payload, label, unit = "" }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={TOOLTIP_STYLE}>
      <div style={{ color: CHART.muted, fontSize: 11, marginBottom: 4 }}>
        {label}{unit}
      </div>
      {payload.filter((p) => p.value != null).map((p) => (
        <div key={p.name} style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <i style={{
            width: 9, height: 9, borderRadius: 2, background: p.color,
            display: "inline-block", flex: "none",
          }} />
          <span style={{ color: CHART.muted }}>{p.name}</span>
          <b style={{ marginLeft: "auto", fontFamily: "var(--mono)" }}>
            {typeof p.value === "number" ? fmt(p.value, p.value % 1 ? 1 : 0) : p.value}
          </b>
        </div>
      ))}
    </div>
  );
}

/** Magnitude on a matrix: one hue, light to dark, with a scale legend. */
function ODMatrix({ od, approaches }) {
  const names = approaches.map((a) => a.name);
  const { m, max } = useMemo(() => {
    const acc = {};
    let mx = 0;
    for (const r of od) {
      const k = `${r.origin}|${r.destination}`;
      acc[k] = (acc[k] || 0) + r.count;
      mx = Math.max(mx, acc[k]);
    }
    return { m: acc, max: mx };
  }, [od]);

  return (
    <>
      <table className="od">
        <thead>
          <tr><th className="corner">from → to</th>{names.map((n) => <th key={n}>{n}</th>)}</tr>
        </thead>
        <tbody>
          {names.map((o) => (
            <tr key={o}>
              <th>{o}</th>
              {names.map((d) => {
                const v = m[`${o}|${d}`] || 0;
                const t = max ? v / max : 0;
                return (
                  <td key={d}
                      style={v
                        ? { background: seqColor(t), color: t > 0.5 ? "#fff" : CHART.ink }
                        : { color: "#3c4553" }}>
                    {v || (o === d ? "·" : "")}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="ramp">
        <span>0</span>
        {[0, 0.2, 0.4, 0.6, 0.8, 1].map((t) => <i key={t} style={{ background: seqColor(t) }} />)}
        <span>{max} veh</span>
      </div>
    </>
  );
}

export default function Insights({ data }) {
  const corridorKeys = Object.keys(data.speed_profiles || {});
  const [corridor, setCorridor] = useState(() => {
    const best = (data.od || []).find(
      (r) => corridorKeys.includes(`${r.origin}->${r.destination}`));
    return best ? `${best.origin}->${best.destination}` : corridorKeys[0] || "";
  });
  const [approach, setApproach] = useState(Object.keys(data.queues || {})[0] || "");

  const profile = data.speed_profiles?.[corridor] || [];
  const fd = data.fundamental?.[corridor] || [];
  const queue = data.queues?.[approach] || [];
  const approaches = data.approaches || [];
  const totals = data.totals || {};

  const turns = useMemo(
    () => (data.od || []).slice(0, 8)
      .map((r) => ({ ...r, label: `${r.origin} → ${r.destination}` })), [data]);

  // Vehicles and PCE are different units, so they cannot share a raw axis.
  // Indexing both to share of total puts them on ONE axis honestly, and the
  // finding - two-wheelers dominate the count but not the road space - becomes
  // the shape of the chart rather than a caption.
  const modal = useMemo(() => {
    const rows = (data.modal_split || []).filter((r) => r.count > 0);
    const nV = rows.reduce((s, r) => s + r.count, 0) || 1;
    const nP = rows.reduce((s, r) => s + (r.pce_total || 0), 0) || 1;
    return rows.map((r) => ({
      mode: r.mode,
      countShare: (100 * r.count) / nV,
      pceShare: (100 * (r.pce_total || 0)) / nP,
    }));
  }, [data]);

  const volumes = useMemo(() => {
    const by = {};
    for (const r of data.approach_volumes || []) {
      if (r.partial) continue;
      by[r.interval_start_s] = { t: r.interval_start_s, ...(by[r.interval_start_s] || {}) };
      by[r.interval_start_s][r.approach] = r.entering;
    }
    return Object.values(by).sort((a, b) => a.t - b.t);
  }, [data]);

  const lanes = useMemo(() => {
    const by = {};
    for (const r of data.lane_volumes || []) {
      const k = `${r.origin} L${r.lane_in}`;
      by[k] = by[k] || { lane: k };
      by[k][r.mode] = (by[k][r.mode] || 0) + r.count;
    }
    return Object.values(by);
  }, [data]);

  const speeding = profile.length
    ? profile.reduce((a, b) => (b.exceeding_share > (a?.exceeding_share ?? -1) ? b : a), null)
    : null;
  const slowest = useMemo(() => {
    // Skip the end bins: they fall in the velocity-window shoulder where speed
    // is not measured reliably, and taking the minimum from them reported
    // "1 km/h at 11 m" for a corridor whose real minimum is mid-junction.
    const inner = profile.slice(1, -1);
    if (!inner.length) return null;
    return inner.reduce((a, b) => (b.median_speed_kmh < a.median_speed_kmh ? b : a));
  }, [profile]);
  const peakQueue = queue.length ? Math.max(...queue.map((q) => q.queue_vehicles)) : null;
  const topMode = modal.length
    ? modal.reduce((a, b) => (b.countShare > a.countShare ? b : a)) : null;

  return (
    <div className="insights">
      <div className="tiles wide">
        <Tile value={fmt(totals.movements_assigned)} label="complete journeys"
              note={`of ${fmt(totals.tracks_moving)} road users tracked`} />
        <Tile value={fmt(approaches.length)} label="approaches"
              note="inferred from trajectories" />
        {topMode && (
          <Tile value={fmt(topMode.countShare)} unit="%" label={`${topMode.mode} share`}
                tone={MODE_COLOR[topMode.mode]}
                note={`but ${fmt(topMode.pceShare)}% of road space`} />
        )}
        {peakQueue != null && (
          <Tile value={fmt(peakQueue)} label={`peak queue · ${approach}`}
                tone={peakQueue > 25 ? STATUS.serious : STATUS.warning}
                note="contiguous vehicles" />
        )}
        {slowest && (
          <Tile value={fmt(slowest.median_speed_kmh)} unit=" km/h" label="slowest point"
                tone={STATUS.warning} note={`at ${fmt(slowest.distance_m)} m along`} />
        )}
      </div>

      <Panel title="Origin–destination"
             sub={`${fmt(totals.movements_assigned)} journeys · shade shows volume`}>
        <ODMatrix od={data.od || []} approaches={approaches} />
      </Panel>

      <Panel title="Turning movements" sub="ranked, coloured by manoeuvre">
        <ResponsiveContainer width="100%" height={236}>
          <BarChart data={turns} layout="vertical"
                    margin={{ left: 6, right: 40, top: 4, bottom: 4 }} barSize={15}>
            <CartesianGrid stroke={CHART.grid} horizontal={false} />
            <XAxis type="number" {...AXIS} />
            <YAxis type="category" dataKey="label" width={104} interval={0} {...AXIS} />
            <Tooltip content={<Tip />} cursor={{ fill: "rgba(255,255,255,.04)" }} />
            <Bar dataKey="count" name="vehicles" radius={BAR_RADIUS_H}
                 label={{ position: "right", fill: CHART.muted, fontSize: 10.5 }}>
              {turns.map((r, i) => <Cell key={i} fill={TURN_COLOR[r.turn] || SERIES[0]} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <div className="legend">
          {["through", "left", "right", "u-turn"].map((t) => (
            <span key={t}><i style={{ background: TURN_COLOR[t] }} />{t}</span>
          ))}
        </div>
      </Panel>

      <Panel title="Modal split"
             sub="share of vehicles vs share of road space — both indexed to 100% on one axis">
        <ResponsiveContainer width="100%" height={236}>
          <BarChart data={modal} margin={{ left: 2, right: 8, top: 8, bottom: 4 }} barGap={2}>
            <CartesianGrid stroke={CHART.grid} vertical={false} />
            <XAxis dataKey="mode" {...AXIS} interval={0} angle={-22} textAnchor="end"
                   height={54} />
            <YAxis width={42} unit="%" {...AXIS} />
            <Tooltip content={<Tip />} cursor={{ fill: "rgba(255,255,255,.04)" }} />
            <Bar dataKey="countShare" name="of vehicles" radius={BAR_RADIUS}>
              {modal.map((r, i) => <Cell key={i} fill={MODE_COLOR[r.mode]} />)}
            </Bar>
            <Bar dataKey="pceShare" name="of road space" fill="#4a5568" radius={BAR_RADIUS} />
          </BarChart>
        </ResponsiveContainer>
        <div className="legend">
          <span><i className="split" />share of vehicles (coloured by mode)</span>
          <span><i style={{ background: "#4a5568" }} />share of road space (PCE)</span>
        </div>
      </Panel>

      <Panel title="Approach volume" sub="vehicles entering per minute" wide>
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart data={volumes} margin={{ left: 2, right: 10, top: 6, bottom: 4 }}>
            <CartesianGrid stroke={CHART.grid} vertical={false} />
            <XAxis dataKey="t" {...AXIS} unit="s" />
            <YAxis width={42} {...AXIS} />
            <Tooltip content={<Tip unit="s" />} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            {approaches.map((a, i) => (
              <Area key={a.name} type="monotone" dataKey={a.name} stackId="1"
                    stroke={SERIES[i % SERIES.length]} strokeWidth={2}
                    fill={SERIES[i % SERIES.length]} fillOpacity={0.2} />
            ))}
          </AreaChart>
        </ResponsiveContainer>
      </Panel>

      {/* Speed and "% over the limit" are different measures. Two plots, one
          axis each - never two scales on a single plot. */}
      <Panel title="Speed along corridor"
             sub={slowest
               ? `median with 15th–85th percentile band · slowest at ${fmt(slowest.distance_m)} m`
               : "median with percentile band"}
             wide>
        <div className="picker">
          {corridorKeys.map((k) => (
            <button key={k} className={k === corridor ? "on" : ""}
                    onClick={() => setCorridor(k)}>{k}</button>
          ))}
        </div>
        <ResponsiveContainer width="100%" height={200}>
          <ComposedChart data={profile} margin={{ left: 2, right: 10, top: 8, bottom: 4 }}>
            <CartesianGrid stroke={CHART.grid} vertical={false} />
            <XAxis dataKey="distance_m" {...AXIS} unit="m" />
            <YAxis width={54} unit=" km/h" {...AXIS} domain={[0, "auto"]} />
            <Tooltip content={<Tip unit=" m" />} />
            <Area dataKey="p85_speed_kmh" name="85th pct" stroke="none"
                  fill={SERIES[0]} fillOpacity={0.18} legendType="none" />
            <Area dataKey="p15_speed_kmh" name="15th pct" stroke="none"
                  fill={CHART.surface} fillOpacity={1} legendType="none" />
            <Line dataKey="median_speed_kmh" name="median" stroke={SERIES[0]}
                  strokeWidth={2.4} dot={false} legendType="none" />
            {slowest && (
              <ReferenceLine x={slowest.distance_m} stroke={STATUS.warning}
                             strokeDasharray="3 3"
                             label={{ value: "slowest", fill: STATUS.warning,
                                      fontSize: 10, position: "insideTopRight" }} />
            )}
          </ComposedChart>
        </ResponsiveContainer>
        <div className="legend">
          <span><i style={{ background: SERIES[0] }} />median speed</span>
          <span><i style={{ background: SERIES[0], opacity: .28 }} />15th–85th percentile</span>
          <span><i style={{ background: STATUS.warning }} />slowest point</span>
        </div>
      </Panel>

      <Panel title="Where speeding concentrates"
             sub={speeding
               ? `peaks at ${fmt(speeding.distance_m)} m — ${fmt(speeding.exceeding_share, 1)}% of samples above 40 km/h`
               : "share of samples above 40 km/h"}>
        <ResponsiveContainer width="100%" height={200}>
          <AreaChart data={profile} margin={{ left: 2, right: 10, top: 8, bottom: 4 }}>
            <CartesianGrid stroke={CHART.grid} vertical={false} />
            <XAxis dataKey="distance_m" {...AXIS} unit="m" />
            <YAxis width={42} unit="%" {...AXIS} domain={[0, "auto"]} />
            <Tooltip content={<Tip unit=" m" />} />
            <Area dataKey="exceeding_share" name="over 40 km/h" type="monotone"
                  stroke={STATUS.critical} strokeWidth={2}
                  fill={STATUS.critical} fillOpacity={0.2} />
          </AreaChart>
        </ResponsiveContainer>
      </Panel>

      <Panel title="Queue length"
             sub={peakQueue != null
               ? `contiguous slow vehicles back from the stop line · peak ${peakQueue}`
               : "contiguous slow vehicles back from the stop line"}>
        <div className="picker">
          {Object.keys(data.queues || {}).map((k) => (
            <button key={k} className={k === approach ? "on" : ""}
                    onClick={() => setApproach(k)}>{k}</button>
          ))}
        </div>
        <ResponsiveContainer width="100%" height={190}>
          <AreaChart data={queue} margin={{ left: 2, right: 10, top: 6, bottom: 4 }}>
            <CartesianGrid stroke={CHART.grid} vertical={false} />
            <XAxis dataKey="t_s" {...AXIS} unit="s" />
            <YAxis width={38} {...AXIS} />
            <Tooltip content={<Tip unit=" s" />} />
            <Area type="stepAfter" dataKey="queue_vehicles" name="vehicles"
                  stroke={SERIES[1]} strokeWidth={1.6}
                  fill={SERIES[1]} fillOpacity={0.22} />
          </AreaChart>
        </ResponsiveContainer>
      </Panel>

      <Panel title="Flow–density" sub="q = k·v on a 60 m segment, 15 s bins">
        <ResponsiveContainer width="100%" height={190}>
          <ScatterChart margin={{ left: 2, right: 12, top: 10, bottom: 4 }}>
            <CartesianGrid stroke={CHART.grid} />
            <XAxis type="number" dataKey="density_veh_per_km" name="density"
                   {...AXIS} unit=" v/km" />
            <YAxis type="number" dataKey="flow_veh_per_h" name="flow" width={56}
                   {...AXIS} unit=" v/h" />
            <Tooltip content={<Tip />} cursor={{ strokeDasharray: "3 3" }} />
            <Scatter data={fd} fill={SERIES[2]} fillOpacity={0.85} />
          </ScatterChart>
        </ResponsiveContainer>
      </Panel>

      <Panel title="Lane volume by mode" sub="who uses which lane" wide>
        <ResponsiveContainer width="100%" height={210}>
          <BarChart data={lanes} margin={{ left: 2, right: 10, top: 6, bottom: 4 }}>
            <CartesianGrid stroke={CHART.grid} vertical={false} />
            <XAxis dataKey="lane" {...AXIS} interval={0} />
            <YAxis width={42} {...AXIS} />
            <Tooltip content={<Tip />} cursor={{ fill: "rgba(255,255,255,.04)" }} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            {MODES.map((m, i) => (
              <Bar key={m} dataKey={m} stackId="a" fill={MODE_COLOR[m]}
                   stroke={CHART.surface} strokeWidth={2}
                   radius={i === MODES.length - 1 ? BAR_RADIUS : 0} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </Panel>
    </div>
  );
}
