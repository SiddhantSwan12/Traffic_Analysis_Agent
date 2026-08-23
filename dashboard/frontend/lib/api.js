// Re-exported from the validated palette so the overlay, the map and every
// chart paint a class the same colour. The previous set was picked to match the
// baked-in video renderer and failed colourblind separation: motorcycle against
// pedestrian scored dE 1.6 under deuteranopia.
export { MODES, MODE_COLOR } from "./viz";

export const STATE = ["moving", "temporarily_stopped", "parked", "unknown"];

export async function getJSON(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path} -> ${r.status}`);
  return r.json();
}

/** Stable per-identity colour: the same golden-angle rotation the renderer uses. */
export function idColor(id) {
  return `hsl(${(id * 137.508) % 360} 85% 60%)`;
}

export function fmtTime(s) {
  if (!isFinite(s)) return "--:--";
  const m = Math.floor(s / 60);
  return `${String(m).padStart(2, "0")}:${(s - m * 60).toFixed(1).padStart(4, "0")}`;
}

/**
 * Fetches overlay chunks on demand and keeps a small window of them.
 *
 * The client drives playback, so it always knows which chunk it needs next —
 * which is exactly why this is plain HTTP rather than a socket. Chunks are
 * immutable and served with a long cache lifetime, so re-watching a segment
 * costs no network at all.
 */
export class ChunkCache {
  constructor(chunkS, nChunks, lookahead = 1) {
    this.chunkS = chunkS;
    this.nChunks = nChunks;
    this.lookahead = lookahead;
    this.map = new Map();
    this.pending = new Map();
  }

  indexFor(timeS) {
    return Math.max(0, Math.min(this.nChunks - 1, Math.floor(timeS / this.chunkS)));
  }

  async ensure(idx) {
    if (idx < 0 || idx >= this.nChunks) return null;
    if (this.map.has(idx)) return this.map.get(idx);
    if (this.pending.has(idx)) return this.pending.get(idx);
    const p = getJSON(`/api/chunk/${idx}`)
      .then((d) => {
        this.map.set(idx, d);
        this.pending.delete(idx);
        // keep memory bounded; the browser cache still holds the bytes
        if (this.map.size > 6) this.map.delete(this.map.keys().next().value);
        return d;
      })
      .catch(() => {
        this.pending.delete(idx);
        return null;
      });
    this.pending.set(idx, p);
    return p;
  }

  prefetch(timeS) {
    const i = this.indexFor(timeS);
    for (let k = 0; k <= this.lookahead; k++) this.ensure(i + k);
  }

  /** Synchronous lookup: returns null if that chunk is not resident yet. */
  frame(frameIdx, fps) {
    const idx = this.indexFor(frameIdx / fps);
    const c = this.map.get(idx);
    if (!c) return null;
    return c[frameIdx] || c[String(frameIdx)] || [];
  }
}
