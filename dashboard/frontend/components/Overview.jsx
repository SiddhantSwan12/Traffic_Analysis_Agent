"use client";

import { useMemo } from "react";
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  AXIS,
  CHART,
  MODE_COLOR,
  SERIES,
  STATUS,
  TOOLTIP_STYLE,
  TURN_COLOR,
  fmt,
} from "@/lib/viz";

function ChartTip({ active, payload, label, suffix = "" }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tip" style={TOOLTIP_STYLE}>
      <span>{label != null ? `T + ${fmt(label)}s` : "Traffic volume"}</span>
      {payload.filter((item) => item.value != null).map((item) => (
        <div key={item.dataKey}>
          <i style={{ background: item.color }} />
          <span>{item.name}</span>
          <b>{fmt(item.value)}{suffix}</b>
        </div>
      ))}
    </div>
  );
}

function Metric({ label, value, unit, note, tone = "blue", progress }) {
  return (
    <article className={`overview-metric ${tone}`}>
      <div className="metric-top">
        <span>{label}</span>
        <i />
      </div>
      <div className="metric-value">{value}<small>{unit}</small></div>
      <p>{note}</p>
      {progress != null && (
        <div className="metric-progress" aria-label={`${label}: ${Math.round(progress)} percent`}>
          <span style={{ width: `${Math.max(0, Math.min(100, progress))}%` }} />
        </div>
      )}
    </article>
  );
}

function Card({ title, sub, action, children, className = "" }) {
  return (
    <section className={`overview-card ${className}`.trim()}>
      <header>
        <div><h2>{title}</h2>{sub && <p>{sub}</p>}</div>
        {action}
      </header>
      <div className="overview-card-body">{children}</div>
    </section>
  );
}

export default function Overview({ data, meta, onNavigate }) {
  const totals = data.totals || {};
  const approaches = data.approaches || [];

  const volumes = useMemo(() => {
    const rows = {};
    for (const item of data.approach_volumes || []) {
      if (item.partial) continue;
      rows[item.interval_start_s] ||= { t: item.interval_start_s, total: 0 };
      rows[item.interval_start_s][item.approach] = item.entering;
      rows[item.interval_start_s].total += item.entering || 0;
    }
    return Object.values(rows).sort((a, b) => a.t - b.t);
  }, [data]);

  const modal = useMemo(() => {
    const rows = (data.modal_split || []).filter((item) => item.count > 0);
    const total = rows.reduce((sum, item) => sum + item.count, 0) || 1;
    return rows
      .map((item) => ({ ...item, share: (100 * item.count) / total }))
      .sort((a, b) => b.count - a.count);
  }, [data]);

  const movements = useMemo(
    () => [...(data.od || [])]
      .sort((a, b) => b.count - a.count)
      .slice(0, 6)
      .map((item) => ({ ...item, label: `${item.origin} → ${item.destination}` })),
    [data],
  );

  const peakQueue = useMemo(() => {
    let peak = null;
    for (const [approach, samples] of Object.entries(data.queues || {})) {
      for (const sample of samples) {
        if (!peak || sample.queue_vehicles > peak.queue_vehicles) peak = { ...sample, approach };
      }
    }
    return peak;
  }, [data]);

  const slowest = useMemo(() => {
    let result = null;
    for (const [corridor, samples] of Object.entries(data.speed_profiles || {})) {
      const trusted = samples.slice(1, -1);
      for (const sample of trusted) {
        if (sample.median_speed_kmh == null) continue;
        if (!result || sample.median_speed_kmh < result.median_speed_kmh) {
          result = { ...sample, corridor };
        }
      }
    }
    return result;
  }, [data]);

  const assignedShare = totals.tracks_moving
    ? (100 * totals.movements_assigned) / totals.tracks_moving
    : 0;
  const peakVolume = volumes.length
    ? volumes.reduce((best, item) => (item.total > best.total ? item : best), volumes[0])
    : null;
  const topMode = modal[0];
  const topMovement = movements[0];

  return (
    <div className="overview-page">
      <section className="overview-hero">
        <div>
          <span className="eyebrow">Intersection performance</span>
          <h2>One clear view of how the network is moving.</h2>
          <p>
            Analysis of <b>{fmt(totals.tracks_moving)}</b> tracked road users across{" "}
            <b>{approaches.length}</b> approaches, derived from the {meta.name.replaceAll("_", " ")} capture.
          </p>
        </div>
        <div className="hero-actions">
          <button className="primary-action" onClick={() => onNavigate("Video")}>
            <PlayIcon /> Inspect video
          </button>
          <button className="secondary-action" onClick={() => onNavigate("Insights")}>
            Open analytics <ArrowIcon />
          </button>
        </div>
      </section>

      <section className="overview-metrics" aria-label="Key traffic metrics">
        <Metric
          label="Road users tracked"
          value={fmt(totals.tracks_moving)}
          note="Moving objects across the full capture"
          tone="blue"
        />
        <Metric
          label="Complete journeys"
          value={fmt(totals.movements_assigned)}
          note={`${fmt(assignedShare, 1)}% assigned origin and destination`}
          progress={assignedShare}
          tone="green"
        />
        <Metric
          label="Peak queue"
          value={peakQueue ? fmt(peakQueue.queue_vehicles) : "—"}
          unit=" vehicles"
          note={peakQueue ? `${peakQueue.approach} at T + ${fmt(peakQueue.t_s)}s` : "No queue samples available"}
          tone="amber"
        />
        <Metric
          label="Lowest median speed"
          value={slowest ? fmt(slowest.median_speed_kmh, 1) : "—"}
          unit=" km/h"
          note={slowest ? `${slowest.corridor} · ${fmt(slowest.distance_m)} m along` : "No speed profile available"}
          tone="violet"
        />
      </section>

      <div className="overview-grid">
        <Card
          title="Traffic volume"
          sub="Vehicles entering each approach by one-minute interval"
          className="volume-card span-8"
          action={<span className="card-badge">{peakVolume ? `Peak ${fmt(peakVolume.total)} / min` : "No samples"}</span>}
        >
          {volumes.length ? (
            <ResponsiveContainer width="100%" height={280}>
              <ComposedChart data={volumes} margin={{ left: 0, right: 12, top: 16, bottom: 0 }}>
                <defs>
                  <linearGradient id="totalVolume" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={SERIES[0]} stopOpacity={0.3} />
                    <stop offset="100%" stopColor={SERIES[0]} stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke={CHART.grid} vertical={false} />
                <XAxis dataKey="t" {...AXIS} axisLine={false} tickFormatter={(value) => `${Math.round(value / 60)}m`} />
                <YAxis width={34} {...AXIS} axisLine={false} />
                <Tooltip content={<ChartTip />} cursor={{ stroke: "rgba(255,255,255,.14)", strokeDasharray: "3 3" }} />
                <Area type="monotone" dataKey="total" name="Total" stroke={SERIES[0]} strokeWidth={2.2} fill="url(#totalVolume)" />
                {approaches.map((approach, index) => (
                  <Line
                    key={approach.name}
                    type="monotone"
                    dataKey={approach.name}
                    name={approach.name}
                    stroke={SERIES[(index + 2) % SERIES.length]}
                    strokeWidth={1.35}
                    strokeOpacity={0.72}
                    dot={false}
                  />
                ))}
              </ComposedChart>
            </ResponsiveContainer>
          ) : <EmptyState>Volume intervals are not available for this capture.</EmptyState>}
          <div className="overview-legend">
            <span><i style={{ background: SERIES[0] }} />All approaches</span>
            {approaches.map((approach, index) => (
              <span key={approach.name}><i style={{ background: SERIES[(index + 2) % SERIES.length] }} />{approach.name}</span>
            ))}
          </div>
        </Card>

        <Card title="Operational readout" sub="The strongest signals in this capture" className="span-4 signal-card">
          <div className="signal-list">
            <Signal
              tone="blue"
              label="Dominant movement"
              value={topMovement?.label || "Not available"}
              note={topMovement ? `${fmt(topMovement.count)} complete journeys` : ""}
            />
            <Signal
              tone="green"
              label="Largest road-user group"
              value={topMode?.mode || "Not available"}
              note={topMode ? `${fmt(topMode.share, 1)}% of tracked traffic` : ""}
            />
            <Signal
              tone="amber"
              label="Queue pressure"
              value={peakQueue ? `${peakQueue.approach} · ${fmt(peakQueue.queue_vehicles)} vehicles` : "Not available"}
              note={peakQueue ? `Peak observed at T + ${fmt(peakQueue.t_s)}s` : ""}
            />
          </div>
          <button className="card-link" onClick={() => onNavigate("Reasoning")}>Review evidence-backed findings <ArrowIcon /></button>
        </Card>

        <Card title="Traffic composition" sub="Share of tracked moving road users" className="span-5">
          <div className="composition-list">
            {modal.map((item) => (
              <div className="composition-row" key={item.mode}>
                <div><span><i style={{ background: MODE_COLOR[item.mode] }} />{item.mode}</span><b>{fmt(item.count)}</b></div>
                <div className="composition-track"><span style={{ width: `${item.share}%`, background: MODE_COLOR[item.mode] }} /></div>
                <small>{fmt(item.share, 1)}%</small>
              </div>
            ))}
          </div>
        </Card>

        <Card title="Top movements" sub="Highest-volume origin–destination pairs" className="span-7">
          {movements.length ? (
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={movements} layout="vertical" margin={{ left: 0, right: 36, top: 2, bottom: 2 }} barSize={17}>
                <CartesianGrid stroke={CHART.grid} horizontal={false} />
                <XAxis type="number" {...AXIS} axisLine={false} />
                <YAxis type="category" dataKey="label" width={96} interval={0} {...AXIS} axisLine={false} />
                <Tooltip content={<ChartTip />} cursor={{ fill: "rgba(255,255,255,.035)" }} />
                <Bar dataKey="count" name="Journeys" radius={[0, 5, 5, 0]} label={{ position: "right", fill: CHART.muted, fontSize: 10 }}>
                  {movements.map((item, index) => (
                    <Cell key={`${item.label}-${index}`} fill={TURN_COLOR[item.turn] || SERIES[index % SERIES.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : <EmptyState>Complete origin–destination movements are not available.</EmptyState>}
        </Card>
      </div>
    </div>
  );
}

function Signal({ tone, label, value, note }) {
  return (
    <div className={`signal ${tone}`}>
      <i />
      <div><span>{label}</span><b>{value}</b>{note && <small>{note}</small>}</div>
    </div>
  );
}

function EmptyState({ children }) {
  return <div className="empty-state">{children}</div>;
}

function PlayIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m9 7 8 5-8 5Z" /></svg>;
}

function ArrowIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="m9 6 6 6-6 6" /></svg>;
}
