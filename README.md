# Drone Traffic Detection & Tracking — FlytBase Visual Hackathon

Detects, classifies and tracks every road user in drone footage, holding a
stable identity through occlusion, crossing paths and long dwell times.
Stationary vehicles are identified and suppressed. Output is an annotated
H.264 video with persistent IDs and motion trails, plus a per-frame CSV.

## What it produces

| Output | Contents |
|---|---|
| `output/<name>_annotated.mp4` | boxes + persistent ID + mode + speed + motion trail + live HUD |
| `output/<name>_tracks.csv` | one row per object per frame: id, mode, box, centre, km/h, length, parked flag |
| `output/<name>_report.json` | unique road users per mode, track-duration stats, scale calibration |
| `output/<name>_detections.parquet` | raw detections, cached so tracking can be re-tuned without re-running YOLO |

## Install

```powershell
pip install numpy pandas scipy opencv-python-headless pyarrow imageio-ffmpeg huggingface_hub ultralytics
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

The `cu128` index matters: an RTX 50-series card is Blackwell (`sm_120`) and the
default PyPI wheels do not carry kernels for it.

## Run

```powershell
# 20-second preview first - always. Judge quality before spending the full run.
python run.py --video Dataset_Video/Intersection_1080p.MP4 --preview 20

# full pass
python run.py --video Dataset_Video/Intersection_1080p.MP4

# re-tune tracking without re-running YOLO (the expensive stage)
python run.py --video Dataset_Video/Intersection_1080p.MP4 --from-detections

# re-render only: trails, labels, parked visibility
python run.py --video Dataset_Video/Intersection_1080p.MP4 --render-only --show-parked
python run.py --video Dataset_Video/Intersection_1080p.MP4 --render-only --labels id
```

The three stages cache to disk, so iteration is cheap: `--from-detections`
skips YOLO (the expensive part, ~8 f/s) and re-runs tracking onward at ~40 f/s;
`--render-only` re-draws from the finished tracks in seconds.

Labels: `--labels full` (`12 motorcycle 34`), `compact` (`12 m/c 34`, default)
or `id` (`12`, class shown by the box's coloured cap). Labels are placed with an
occupancy grid — bigger objects claim space first, and a label that cannot fit
without overlapping degrades to the bare ID and then to nothing, so dense
clusters stay readable instead of turning into a pile of text.

Verify the logic without a GPU or the real video:

```powershell
python tests/test_pipeline.py
```

## The 7 classes

The detector is trained on VisDrone's 10 labels; the brief asks for 7 transport
modes. LGV and HGV are *weight* categories that appearance alone cannot
settle from the air, so they are resolved by physical length.

| VisDrone | mode | then refined by length |
|---|---|---|
| `pedestrian`, `people` | pedestrian | — |
| `bicycle`, `motor`, `tricycle`, `awning-tricycle` | motorcycle | — |
| `car` | car | > 6.2 m → LGV |
| `van` | LGV | > 7.5 m → truck, > 9.5 m → HGV |
| `truck` | truck | > 9.5 m → HGV |
| `bus` | bus | < 7.0 m → LGV |

Length comes from a **metres-per-pixel model calibrated against cars**. The
gimbal sits at about −63°, so scale varies down the frame; the model fits
apparent car length against image row and inverts it. Using `max(width, height)`
of the axis-aligned box is deliberate — for a 4.3 × 1.8 m rectangle that
quantity stays within ~1% of 4.3 m at *every* rotation angle, which makes it an
unusually stable ruler. The same model gives speed in km/h.

Class is decided **per track, not per frame**: a confidence-weighted vote over
the object's entire history. This is what stops the label flickering
car → van → truck between frames.

## How the hard requirements are met

**Stable identity through occlusion.** Three layers. Unmatched tracks keep being
Kalman-predicted and stay eligible for matching for 90 frames (3 s). Beyond
that they move to a long-term gallery — still predicted forward, so a vehicle
that reappears 600 px from where it vanished is matched against where it
*should* be, not where it was last seen — and are recovered on appearance plus a
25 m spatial gate for up to 20 s. Finally, gaps of ≤ 30 frames are filled by
interpolation so boxes never blink.

**Stable identity through crossing paths.** Association fuses IoU with a 124-D
colour/layout descriptor (6×6 BGR grid + saturation-weighted hue histogram,
EMA-updated per track), and adds a soft penalty for matching across transport
modes. Two vehicles that overlap for a few frames stay distinguishable by
appearance even when geometry is ambiguous.

**Stable identity through long dwell.** A vehicle halted at a signal is still
detected every frame, so it stays matched by IoU while its Kalman velocity
decays to zero. No timeout applies while it remains visible.

**Camera-motion compensation.** Even in a stationary hover the gimbal
micro-corrects. Left uncompensated, the Kalman filter absorbs that drift as
object velocity and mispredicts *every* box at once — the single biggest source
of ID switches in dense top-down scenes. A RANSAC affine fit over background
optical flow (detections masked out) removes it each frame.

## Stationary vehicles

A track is flagged `parked` and hidden only when **all four** hold: it has lived
≥ 3 s, moved above 0.7 m/s in < 5% of frames, has net displacement < 2.5 m, and
total path length < 6 m.

Requiring all four is what separates *parked* from *waiting*. A car stopped at a
red light for 90 s accumulates a large path length before and after the stop, so
it stays visible with its identity intact — exactly what the brief asks for.

Both motion measurements deliberately avoid frame-to-frame differencing. Box
jitter is zero-mean, but taking `|difference|` first and averaging afterwards
*rectifies* that noise into a positive bias, which makes a perfectly static
object appear to crawl at a few km/h and slowly accumulate metres of phantom
travel. Speed is therefore measured as displacement across a ±7-frame window,
and path length over centres resampled every 0.5 s. Both cancel the jitter
instead of accumulating it.

This also removes the building-facade false positives **automatically**: a
detector hallucinating cars on a rooftop produces perfectly static tracks, which
satisfy all four conditions. No hand-drawn exclusion rectangle is needed, and
nothing has to be recalibrated per camera angle. `--show-parked` reveals them if
you want to inspect what was suppressed.

## Why detection changed

The earlier approach ran SAHI sliced inference on every 5th frame, then held
stale boxes for the intervening frames — which is what produced the visible lag
and flicker. Two changes:

- **Every frame is processed.** Tracking quality depends on dense temporal
  sampling far more than on detector strength.
- **Tiles are batched into one forward pass** instead of SAHI's sequential
  slicing, and duplicates are removed by geometry rather than by NMS
  heuristics. Each tile owns an exclusive "core" region; a detection is kept
  only from the tile whose core contains its centre. Because the 200 px overlap
  exceeds the largest object, an object centred in a core is *provably* whole
  within that tile. The cores partition the frame exactly once — asserted in
  the test suite.

Detector confidence is deliberately low (0.15). Recall is cheap here and the
tracker is the filter: weak detections feed ByteTrack's second association
stage, which is what keeps a partially-occluded vehicle alive.

## Tuning

Everything lives in `vtrack/config.py`.

| Symptom | Knob |
|---|---|
| Missing small objects | `detect.conf` ↓, or `detect.model` → `mshamrai/yolov8x-visdrone` |
| Too many false positives | `post.min_track_len` ↑, `post.min_track_hits` ↑ |
| IDs switch when vehicles cross | `track.w_appearance` ↑, `track.app_gate` ↓ |
| IDs lost through long occlusion | `track.reid_max_age` ↑, `track.reid_dist` ↑ |
| Waiting vehicles wrongly hidden | `post.parked_path_len_m` ↓, `post.parked_moving_frac` ↑ |
| Parked cars still showing | `post.parked_net_disp_m` ↑ |
| Boxes look jittery | `post.smooth_window` ↑ |
| Trails too long/short | `render.trail_frames` |

## Layout

```
run.py                 CLI: detect -> track -> post-process -> render
vtrack/config.py       every tunable, in one place
vtrack/detector.py     batched tiled YOLO, core-region deduplication
vtrack/tracker.py      association, appearance, ReID gallery
vtrack/kalman.py       8-state constant-velocity filter
vtrack/cmc.py          camera-motion compensation
vtrack/postprocess.py  interpolation, smoothing, class vote, speed, parked
vtrack/taxonomy.py     VisDrone -> 7 modes, metric scale model
vtrack/telemetry.py    DJI .srt parsing
vtrack/render.py       annotated video, piped straight to ffmpeg
tests/test_pipeline.py synthetic checks, no GPU required
```

Rendering pipes raw frames into ffmpeg over stdin, so the multi-gigabyte
intermediate file the earlier version wrote is never created. With
`h264_nvenc` the encode is GPU-accelerated.
