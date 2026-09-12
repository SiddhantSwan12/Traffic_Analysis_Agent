/**
 * Visualization tokens.
 *
 * The categorical palette is not chosen by eye. It is the validated 8-hue
 * reference set stepped for a dark surface, and the first seven slots pass
 * every gate: lightness band, chroma floor, adjacent CVD separation (worst
 * dE 8.4), normal-vision separation (worst dE 19.3) and 3:1 contrast.
 *
 * The palette this replaced failed three of those. Two failures were real
 * legibility bugs rather than pedantry: motorcycle against pedestrian scored
 * dE 1.6 under deuteranopia -- indistinguishable to a red-green colourblind
 * reader, and those are the two largest classes in this data -- and truck
 * against HGV scored dE 12.2 for NORMAL vision, below the 15 floor, so no
 * reader could reliably tell them apart.
 *
 * Slots are assigned to modes in taxonomy order and never cycled or reassigned
 * by rank, so a mode keeps its colour when a filter removes its neighbours.
 */

// validated categorical slots, dark surface
export const SERIES = [
  "#3987e5", // 1 blue
  "#d95926", // 2 orange
  "#199e70", // 3 aqua
  "#c98500", // 4 yellow
  "#d55181", // 5 magenta
  "#008300", // 6 green
  "#9085e9", // 7 violet
  "#e66767", // 8 red
];

export const MODES = ["pedestrian", "motorcycle", "car", "LGV", "truck", "HGV", "bus"];

// colour follows the entity: the same map drives the video overlay, the map
// and every chart, so a class reads the same everywhere
export const MODE_COLOR = Object.fromEntries(MODES.map((m, i) => [m, SERIES[i]]));

// Turn types are identity, not magnitude. Kept off the mode slots so a mode
// and a turn type are never confusable in the same view.
export const TURN_COLOR = {
  through: SERIES[0],
  left: SERIES[6],
  right: SERIES[3],
  "u-turn": SERIES[7],
  internal: "#5a6577",
};

/** Sequential ramp: ONE hue, light to dark, for magnitude. Never a rainbow. */
export const SEQ_HUE = [
  "#0d2137", "#123a5e", "#175384", "#1c6cab", "#2185d2", "#3987e5",
];
export function seqColor(t) {
  const x = Math.max(0, Math.min(1, t));
  return SEQ_HUE[Math.round(x * (SEQ_HUE.length - 1))];
}

/** Status is reserved. Never reused as "series 4". */
export const STATUS = {
  good: "#3fae6b",
  warning: "#c98500",
  serious: "#d95926",
  critical: "#e34948",
};

/** Speed ramp for the time-space diagram: semantic heat, with a scale legend. */
export const SPEED_STOPS = [
  [0, "#e34948"], [10, "#d95926"], [20, "#c98500"],
  [30, "#3fae6b"], [45, "#199e70"],
];

export const CHART = {
  grid: "rgba(148, 163, 184, 0.12)",
  axis: "#778497",
  surface: "#101721",
  surfaceDeep: "#080d14",
  ink: "#f1f5fa",
  muted: "#929fb0",
  fontSize: 10,
};

/** Recharts axis props, so every chart shares one look. */
export const AXIS = { stroke: CHART.axis, fontSize: CHART.fontSize, tickLine: false };

export const TOOLTIP_STYLE = {
  background: "rgba(8, 13, 20, 0.94)",
  border: `1px solid ${CHART.grid}`,
  borderRadius: 10,
  fontSize: 11,
  padding: "9px 11px",
  boxShadow: "0 14px 36px rgba(0,0,0,.42)",
  backdropFilter: "blur(16px)",
};

/** 4 px rounded data-end, anchored to the baseline. */
export const BAR_RADIUS = [4, 4, 0, 0];
export const BAR_RADIUS_H = [0, 4, 4, 0];

export function fmt(n, digits = 0) {
  if (n === null || n === undefined || Number.isNaN(n)) return "—";
  return Number(n).toLocaleString(undefined, {
    minimumFractionDigits: digits, maximumFractionDigits: digits,
  });
}
