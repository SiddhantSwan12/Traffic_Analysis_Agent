# Build Prompt — Drone Traffic Detection & Tracking Pipeline

A self-contained specification of the system in this repository, written as a
prompt. Handed to a capable engineer or coding agent, it should reproduce the
pipeline.

---

## Task

Build a Python pipeline that processes drone-captured traffic video and produces
an annotated video plus a per-frame CSV. It must detect every road user,
classify it into one of seven transport modes, track it with a stable identity,
suppress stationary vehicles, and draw motion trails and persistent IDs.

**Input:** 1920×1080 H.264 video from a drone hovering at ~70 m AGL with the
gimbal at ~−63° (oblique, not nadir), 29.97 fps, ~12,000 frames, with an
optional DJI `.srt` telemetry sidecar.

**Target hardware:** a single consumer GPU (8 GB). The full video must process
in well under an hour.

---

## Hard requirements

1. **Seven-class classification** — `car`, `LGV`, `HGV`, `bus`, `truck`,
   `motorcycle`, `pedestrian`.
2. **Stable identity** held through occlusion, crossing paths, and long dwell
   times.
3. **Stationary vehicles must not be shown.** Parked cars are excluded. A
   vehicle merely *waiting* at a signal must still be shown, with its identity
   intact.
4. **Motion trails** behind moving objects, and a **persistent visible ID** on
   every tracked object.

---

## Constraints and design rulings

### Detection

- Use a VisDrone-finetuned YOLO model (e.g. `mshamrai/yolov8m-visdrone` from the
  HF Hub). Resolve the weight filename by listing the repo rather than
  hardcoding it. Read class names from the model, never from a literal.
- **Process every frame.** Do not sample every Nth frame and hold stale boxes —
  that produces visible lag and flicker, and starves the tracker of the dense
  temporal sampling it depends on.
- Objects are 11–70 px, so a single full-frame pass loses them. Tile the frame,
  but **batch all tiles into one forward pass** rather than running them
  sequentially (as SAHI does).
- **Deduplicate tiles by geometry, not by NMS.** Give each tile an exclusive
  "core" region, taken as the midpoints of consecutive overlaps; keep a
  detection only from the tile whose core contains its centre. Assert that the
  overlap exceeds the largest expected object — then an object centred in a core
  is *provably* whole within that tile, and cross-tile duplicates are impossible
  by construction. Write a test that the cores partition the frame exactly once.
- Make tiling scale with frame width so 4K input works without reconfiguration.
- Set detector confidence **low** (~0.15). Recall is cheap; the tracker is the
  filter. Weak detections feed the second association stage, which is what keeps
  a partially-occluded vehicle alive.
- Decode frames on a background thread so the GPU never waits on the decoder.

### Class taxonomy

The detector emits VisDrone's 10 labels; the brief asks for 7 modes. Map:

| VisDrone | mode |
|---|---|
| `pedestrian`, `people` | pedestrian |
| `bicycle`, `motor`, `tricycle`, `awning-tricycle` | motorcycle |
| `car` | car |
| `van` | LGV |
| `truck` | truck |
| `bus` | bus |

LGV and HGV are **weight** categories that appearance alone cannot settle from
the air. Resolve them from physical length:

- Calibrate **metres-per-pixel from the apparent size of cars**. Because the
  view is oblique, scale varies down the frame — fit apparent car length against
  image row and invert it, rather than assuming one global scalar.
- Use `max(width, height)` of the axis-aligned box as the length estimate. For a
  4.3 × 1.8 m rectangle this quantity stays within ~1% of 4.3 m at *every*
  rotation angle, which makes it an unusually stable ruler.
- Then refine: car > 6.2 m → LGV; LGV > 7.5 m → truck; > 9.5 m → HGV; bus
  < 7.0 m → LGV.

**Decide class per track, not per frame** — a confidence-weighted vote over the
object's whole history. This is what stops the label flickering car → van →
truck between frames.

### Tracking

Implement the tracker directly rather than depending on an off-the-shelf library
whose API churns. Required parts:

- **Kalman filter**, 8-state constant velocity on `(cx, cy, aspect, height)`.
- **Camera-motion compensation.** Even in a stationary hover the gimbal
  micro-corrects. Left uncompensated, the filter absorbs that drift as object
  velocity and mispredicts *every* box at once — the single biggest source of ID
  switches in dense top-down scenes. Estimate a RANSAC affine fit over
  background optical flow each frame, with detections masked out, and warp all
  track states by it.
- **Two-stage (ByteTrack-style) association**: high-confidence detections first,
  then a second pass over low-confidence ones for tracks still unmatched.
- **Appearance descriptor** fused into stage-1 cost: a cheap 124-D vector
  (6×6 BGR layout grid + 16-bin saturation-weighted hue histogram, L2
  normalised), EMA-updated per track. Add a soft penalty for matching across
  transport modes. This is what prevents ID swaps when two vehicles cross.
- **Long-term re-identification gallery.** Tracks unmatched beyond `max_age`
  (~90 frames) move to a gallery kept for ~20 s, matched on appearance plus a
  spatial gate. **Keep predicting gallery tracks forward** (position
  extrapolates, box shape frozen) — a vehicle hidden for four seconds is not
  where it vanished, and matching against a stale position fails the gate on
  reappearance.
- Solve each association with the Hungarian algorithm; reject pairs above a cost
  threshold.
- Long dwell needs no special case: a stopped vehicle is still detected every
  frame, so it stays matched by IoU while its velocity decays to zero.

### Post-processing (after tracking — these need whole-track history)

- Drop tracks that are too short or have too few real observations (flicker
  suppression).
- Interpolate gaps ≤ 30 frames so boxes do not blink during brief occlusion.
- Smooth box coordinates with a centred moving average, per contiguous segment.
- **Measure motion without rectifying noise.** Box jitter is zero-mean, but
  computing `mean(|frame-to-frame difference|)` takes the magnitude *first* and
  so biases it positive — a perfectly static object then appears to crawl at a
  few km/h and accumulates metres of phantom travel over a long track. Instead:
  - speed = displacement across a **±7-frame window**;
  - path length = sum over centres **resampled every 0.5 s**.
- **Parked classification.** Flag a track parked only when *all four* hold: it
  has lived ≥ 3 s, exceeded 0.7 m/s in < 5% of frames, net displacement
  < 2.5 m, and total path length < 6 m.
  Requiring all four is what separates *parked* from *waiting*: a car stopped at
  a light for 90 s still accumulates a large path length before and after the
  stop, so it stays visible.
  This also removes static false positives (a detector hallucinating vehicles on
  a rooftop produces perfectly static tracks) **automatically** — so no
  hand-drawn exclusion rectangle is needed and nothing must be recalibrated per
  camera angle. Do not use a manual exclusion mask as the primary mechanism; it
  deletes real vehicles and does not transfer between camera framings.
- Assign compact display IDs ordered by first appearance among moving tracks.

### Rendering

- Draw box, persistent ID, mode, speed in km/h, and a fading motion trail
  (~2 s of history) coloured per identity. Give each box a thin mode-coloured
  cap so class reads even when the text label is shortened.
- **Place labels with an occupancy grid.** Larger objects claim space first; a
  label that will not fit without overlapping degrades to the bare ID and then
  to nothing. Dense clusters must stay readable rather than becoming a pile of
  overlapping text.
- HUD panel: timestamp, frame number, and per-mode live and cumulative counts.
- Hide parked tracks by default, with a flag to reveal them for inspection.
- **Pipe raw frames straight into ffmpeg over stdin.** Do not write a
  multi-gigabyte intermediate file. Prefer the GPU encoder (`h264_nvenc`) when
  available, falling back to `libx264`.

### Engineering

- Three stages, each cached to disk, so iteration is cheap: re-running tracking
  without re-running YOLO, and re-rendering without re-tracking, must both be
  one flag.
- All tunables in a single config module.
- Ship a **synthetic test suite that needs no GPU and no real video**. It must
  construct a scene exercising: two objects crossing head-on at nearly the same
  image row; an object hidden for ~20 frames (interpolation); an object hidden
  for >`max_age` frames and reappearing far away (gallery re-ID); a parked
  object; and a static false positive. Assert one stable ID per object, correct
  parked flags, and sane speeds. Also assert the tile cores partition the frame.
- Always run a short preview before committing to a full pass, and inspect
  rendered frames rather than trusting the summary numbers.

---

## Deliverables

| File | Contents |
|---|---|
| `<name>_annotated.mp4` | 1080p H.264: boxes, IDs, mode, speed, trails, HUD |
| `<name>_tracks.csv` | per object per frame: id, mode, box, centre, km/h, length_m, parked, interpolated |
| `<name>_report.json` | unique road users per mode, track-duration stats, scale calibration |
| `<name>_detections.parquet` | raw detections, cached for re-tuning |

---

## Anti-goals

- Do not sample sparsely and hold stale boxes between samples.
- Do not rely on a hand-drawn exclusion rectangle to suppress false positives.
- Do not decide class per frame.
- Do not measure speed by frame-to-frame differencing.
- Do not write a large intermediate video file before encoding.
- Do not report success without inspecting rendered output.
