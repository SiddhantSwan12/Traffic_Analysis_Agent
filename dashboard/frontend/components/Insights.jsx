"use client";

import { useMemo, useState } from "react";
import {
  ResponsiveContainer, LineChart, Line, BarChart, Bar, ScatterChart, Scatter,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, Cell, AreaChart, Area,
} from "recharts";
import { MODE_COLOR, MODES } from "@/lib/api";

const AX = { stroke: "#8a94a6", fontSize: 11 };
const GRID = "#232a36";

function Panel({ title, sub, children, wide }) {
  return (
    <section className={`panel${wide ? " wide" : ""}`}>
      <header><h3>{title}</h3>{sub && <p>{sub}</p>}</header>
      <div className="panel-body">{children}</div>
    </section>
  );
}

/** Origin-destination matrix as a heat grid. */
function ODMatrix({ od, approaches }) {
  const names = approaches.map((a) => a.name);
  const grid = useMemo(() => {
    const m = {};
    let max = 0;
    for (const r of od) {
      m[`${r.origin}|${r.destination}`] = (m[`${r.origin}|${r.destination}`] || 0) + r.count;
      max = Math.max(max, m[`${r.origin}|${r.destination}`]);
    }
    return { m, max };
  }, [od]);

  return (
    <table className="od">
      <thead>
        <tr><th className="corner">from \ to</th>{names.map((n) => <th key={n}>{n}</th>)}</tr>
      </thead>
      <tbody>
        {names.map((o) => (
          <tr key={o}>
            <th>{o}</th>
            {names.map((d) => {
              const v = grid.m[`${o}|${d}`] || 0;
              const a = grid.max ? v / grid.max : 0;
              return (
                <td key={d} style={{ background: v ? `rgba(80,170,255,${0.12 + a * 0.75})` : "transparent" }}>
                  {v || (o === d ? "-" : "")}
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function Insights({ data }) {
  const [corridor, setCorridor] = useState(() => {
    // the dominant movement is the useful default; alphabetical order is not
    const keys = Object.keys(data.speed_profiles || {});
    const best = (data.od || []).find((r) => keys.includes(`${r.origin}->${r.destination}`));
    return best ? `${best.origin}->${best.destination}` : keys[0] || "";
  });
  const [approach, setApproach] = useState(
    Object.keys(data.queues || {})[0] || "");

  const modal = (data.modal_split || []).filter((r) => r.count > 0);
  // Recharts maps a category axis by field value; a function dataKey silently
  // mis-pairs ticks with bars, so the label is materialised into the row.
  const turns = useMemo(
    () => (data.od || []).slice(0, 8).map((r) => ({ ...r, label: `${r.origin}->${r.destination}` })),
    [data]);
  const profile = data.speed_profiles?.[corridor] || [];
  const fd = data.fundamental?.[corridor] || [];
  const queue = data.queues?.[approach] || [];

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

  const approaches = data.approaches || [];
  const speeding = profile.length
    ? profile.reduce((a, b) => (b.exceeding_share > (a?.exceeding_share ?? -1) ? b : a), null)
    : null;

  return (
    <div className="insights">
      <Panel title="Origin–destination" sub={`${data.totals?.movements_assigned ?? 0} complete journeys`}>
        <ODMatrix od={data.od || []} approaches={approaches} />
      </Panel>

      <Panel title="Turning movements" sub="share of all assigned journeys">
        <ResponsiveContainer width="100%" height={230}>
          <BarChart data={turns} layout="vertical"
                    margin={{ left: 4, right: 12, top: 4, bottom: 4 }}>
            <CartesianGrid stroke={GRID} horizontal={false} />
            <XAxis type="number" {...AX} />
            <YAxis type="category" dataKey="label" width={92} interval={0} {...AX} />
            <Tooltip contentStyle={{ background: "#151a23", border: `1px solid ${GRID}` }}
                     formatter={(v, _n, p) => [`${v} veh (${p.payload.share_pct}%)`, p.payload.turn]} />
            <Bar dataKey="count" radius={[0, 3, 3, 0]}>
              {turns.map((r, i) => (
                <Cell key={i} fill={{ through: "#4aa3ff", left: "#c778ff",
                                      right: "#ffa04a", "u-turn": "#ff5a5a" }[r.turn] || "#4aa3ff"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Panel>

      <Panel title="Modal split" sub="vehicles vs road space they occupy (PCE)">
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={modal} margin={{ left: 4, right: 8, top: 4, bottom: 4 }}>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="mode" {...AX} />
            <YAxis width={46} {...AX} />
            <Tooltip contentStyle={{ background: "#151a23", border: `1px solid ${GRID}` }} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Bar dataKey="count" name="vehicles" radius={[3, 3, 0, 0]}>
              {modal.map((r, i) => <Cell key={i} fill={MODE_COLOR[r.mode]} />)}
            </Bar>
            <Bar dataKey="pce_total" name="PCE" fill="#5a6577" radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </Panel>

      <Panel title="Approach volume" sub="vehicles entering per minute" wide>
        <ResponsiveContainer width="100%" height={190}>
          <AreaChart data={volumes} margin={{ left: 4, right: 8, top: 4, bottom: 4 }}>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="t" {...AX} unit="s" />
            <YAxis width={46} {...AX} />
            <Tooltip contentStyle={{ background: "#151a23", border: `1px solid ${GRID}` }} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            {approaches.map((a, i) => (
              <Area key={a.name} type="monotone" dataKey={a.name} stackId="1"
                    stroke={["#4aa3ff", "#8cff5a", "#ffa04a"][i % 3]}
                    fill={["#4aa3ff", "#8cff5a", "#ffa04a"][i % 3]} fillOpacity={0.22} />
            ))}
          </AreaChart>
        </ResponsiveContainer>
      </Panel>

      <Panel
        title="Speed profile along corridor"
        sub={speeding ? `speeding peaks at ${Math.round(speeding.distance_m)} m — ${speeding.exceeding_share}% above 40 km/h`
                      : "median with 15th–85th percentile band"}
        wide
      >
        <div className="picker">
          {Object.keys(data.speed_profiles || {}).map((k) => (
            <button key={k} className={k === corridor ? "on" : ""}
                    onClick={() => setCorridor(k)}>{k}</button>
          ))}
        </div>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={profile} margin={{ left: 4, right: 8, top: 4, bottom: 4 }}>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="distance_m" {...AX} unit="m" />
            <YAxis yAxisId="v" width={56} {...AX} unit=" km/h" domain={[0, "auto"]} />
            <YAxis yAxisId="p" orientation="right" width={44} {...AX} unit="%" domain={[0, 100]} />
            <Tooltip contentStyle={{ background: "#151a23", border: `1px solid ${GRID}` }} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Line yAxisId="v" dataKey="p85_speed_kmh" name="p85" stroke="#3d5570" dot={false} strokeWidth={1} />
            <Line yAxisId="v" dataKey="median_speed_kmh" name="median" stroke="#4aa3ff" dot={false} strokeWidth={2.5} />
            <Line yAxisId="v" dataKey="p15_speed_kmh" name="p15" stroke="#3d5570" dot={false} strokeWidth={1} />
            <Line yAxisId="p" dataKey="exceeding_share" name="% over 40" stroke="#ff5a5a"
                  dot={false} strokeDasharray="4 3" />
          </LineChart>
        </ResponsiveContainer>
      </Panel>

      <Panel title="Queue length" sub="contiguous slow vehicles back from the stop line">
        <div className="picker">
          {Object.keys(data.queues || {}).map((k) => (
            <button key={k} className={k === approach ? "on" : ""}
                    onClick={() => setApproach(k)}>{k}</button>
          ))}
        </div>
        <ResponsiveContainer width="100%" height={185}>
          <AreaChart data={queue} margin={{ left: 4, right: 8, top: 4, bottom: 4 }}>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="t_s" {...AX} unit="s" />
            <YAxis width={46} {...AX} />
            <Tooltip contentStyle={{ background: "#151a23", border: `1px solid ${GRID}` }} />
            <Area type="stepAfter" dataKey="queue_vehicles" name="vehicles"
                  stroke="#ffa04a" fill="#ffa04a" fillOpacity={0.25} />
          </AreaChart>
        </ResponsiveContainer>
      </Panel>

      <Panel title="Flow–density" sub="q = k·v on a 60 m segment, 15 s bins">
        <ResponsiveContainer width="100%" height={185}>
          <ScatterChart margin={{ left: 4, right: 8, top: 8, bottom: 4 }}>
            <CartesianGrid stroke={GRID} />
            <XAxis type="number" dataKey="density_veh_per_km" name="density"
                   {...AX} unit=" v/km" />
            <YAxis type="number" dataKey="flow_veh_per_h" name="flow" width={60} {...AX} unit=" v/h" />
            <Tooltip contentStyle={{ background: "#151a23", border: `1px solid ${GRID}` }}
                     cursor={{ strokeDasharray: "3 3" }} />
            <Scatter data={fd} fill="#8cff5a" />
          </ScatterChart>
        </ResponsiveContainer>
      </Panel>

      <Panel title="Lane volume by mode" sub="who uses which lane" wide>
        <ResponsiveContainer width="100%" height={190}>
          <BarChart data={lanes} margin={{ left: 4, right: 8, top: 4, bottom: 4 }}>
            <CartesianGrid stroke={GRID} vertical={false} />
            <XAxis dataKey="lane" {...AX} />
            <YAxis width={46} {...AX} />
            <Tooltip contentStyle={{ background: "#151a23", border: `1px solid ${GRID}` }} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            {MODES.map((m) => (
              <Bar key={m} dataKey={m} stackId="a" fill={MODE_COLOR[m]} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </Panel>
    </div>
  );
}
