"""Central configuration. Every stage reads from here; the CLI can override."""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


@dataclass
class DetectCfg:
    # HF repo id, or a local path to a .pt file.
    #   mshamrai/yolov8s-visdrone   - fastest, weakest
    #   mshamrai/yolov8m-visdrone   - default, good speed/quality balance
    #   mshamrai/yolov8x-visdrone   - slowest, strongest
    #   dronefreak/visdrone-yolov11m- newer backbone, worth trying
    model: str = "mshamrai/yolov8m-visdrone"
    device: str = "cuda:0"
    half: bool = True

    # Tiled inference. Objects are 11-70 px at 70 m AGL, so a single full-frame
    # pass loses them. Tiles are batched into ONE forward pass (unlike SAHI,
    # which runs them sequentially) -- same recall, several times faster.
    tile: int = 640          # square tile edge, fed to the net at imgsz=tile
    tile_overlap: int = 200  # px; MUST exceed the largest object's extent
    max_object_px: int = 160 # sanity bound used to validate tile_overlap

    conf: float = 0.15       # deliberately low: the tracker filters, not the detector
    nms_iou: float = 0.60
    stride: int = 1          # 1 = every frame. Tracking quality depends on this.
    reader_queue: int = 8    # frames buffered by the decode thread


@dataclass
class TrackCfg:
    high_thresh: float = 0.45      # ByteTrack stage-1 detections
    low_thresh: float = 0.15       # ByteTrack stage-2 (recovers occluded objects)
    init_thresh: float = 0.55      # confidence needed to spawn a new track
    min_hits: int = 3              # frames before a track is "confirmed"

    match_iou: float = 0.80        # stage-1 max cost (1 - fused similarity)
    low_match_iou: float = 0.50
    unconfirmed_iou: float = 0.70

    w_appearance: float = 0.35     # blend weight of appearance vs IoU in stage 1
    app_gate: float = 0.55         # max appearance distance for any match

    max_age: int = 90              # frames a track survives unmatched (3 s @30fps)
    reid_max_age: int = 600        # long-term gallery (20 s) for occlusion recovery
    reid_dist: float = 0.32        # appearance distance to resurrect from gallery
    reid_radius_m: float = 25.0    # spatial gate on resurrection

    use_cmc: bool = True           # compensate gimbal drift / micro-yaw
    cmc_downscale: int = 4


@dataclass
class KinematicsCfg:
    # Velocity from a centred temporal baseline. "centered_diff" is the
    # endpoint difference the L2 spec asks for; "savgol" is the least-squares
    # slope over the same span, which is ~1.7x less noisy because it uses every
    # sample in the window rather than just the two ends.
    velocity_method: str = "centered_diff"   # centered_diff | savgol
    velocity_window_s: float = 0.47          # ~ +/-7 frames at 29.97 fps
    acceleration_window_s: float = 0.87      # ~ +/-13 frames; wider than velocity

    min_speed_for_heading_mps: float = 1.0   # below this, direction is noise
    max_plausible_speed_mps: float = 55.0    # ~200 km/h; above = measurement failure
    # Emergency braking on dry asphalt is about -8 m/s2. Beyond this bound a
    # reading is a measurement failure (box flip, bad interpolation), not a
    # vehicle. Such rows are nulled and flagged, never clamped into a
    # believable-looking number.
    max_plausible_accel_mps2: float = 10.0
    # Acceleration is a second derivative, so it is far more sensitive to
    # invented data than speed is. Once half the window is interpolated the
    # result describes the interpolator, not the vehicle.
    max_interp_fraction: float = 0.5

    reference_car_length_m: float = 4.3
    calibration: str = "auto"                # auto | homography | row_scale | none


@dataclass
class MotionCfg:
    moving_speed_mps: float = 0.7            # ~2.5 km/h
    parked_min_life_s: float = 3.0
    parked_moving_frac: float = 0.05
    parked_net_disp_m: float = 2.5
    # Duration-INDEPENDENT test. A total-path-length budget cannot work: path
    # length grows with track lifetime, so residual jitter alone accumulated
    # 55-67 m over a 400 s stationary vehicle and made the condition
    # unsatisfiable for exactly the parked cars it was meant to catch.
    parked_window_s: float = 10.0
    parked_window_disp_m: float = 3.0


@dataclass
class PostCfg:
    min_track_len: int = 15        # frames; kills flicker false positives
    min_track_hits: int = 8        # real detections (not interpolated)
    max_gap_interp: int = 30       # linearly fill occlusion gaps up to this long
    smooth_window: int = 9         # centered moving average on box coords
    speed_smooth: int = 15

    car_length_m: float = 4.3      # calibration reference for metres-per-pixel

    # Stationary / parked classification (all must hold)
    moving_speed_mps: float = 0.7  # ~2.5 km/h
    parked_moving_frac: float = 0.05
    parked_net_disp_m: float = 2.5
    parked_min_life_s: float = 3.0
    # Largest distance covered in ANY window of this length. Duration
    # independent by construction: a bound on total path length cannot work,
    # because residual jitter accumulates without limit as a track gets longer,
    # so a 400 s parked car would fail any fixed path budget.
    parked_window_s: float = 10.0
    parked_window_disp_m: float = 3.0


@dataclass
class RenderCfg:
    trail_frames: int = 60         # 2 s of history
    trail_thickness: int = 3
    box_thickness: int = 2
    show_parked: bool = False
    show_speed: bool = True
    show_hud: bool = True
    # full    -> "12 motorcycle 34"   (readable, but overlaps badly in dense scenes)
    # compact -> "12 m/c 34"          (default)
    # id      -> "12"                 (class conveyed by the box colour cap)
    label_mode: str = "compact"
    font_scale: float = 0.36
    label_min_box_px: int = 26     # objects smaller than this get an id only
    show_accel: bool = False       # append signed longitudinal accel to labels
    dim_stopped: bool = True       # draw temporarily-stopped objects dimmer
    crf: int = 20
    encoder: str = "auto"          # auto | h264_nvenc | libx264


@dataclass
class Config:
    video: Path = Path("Dataset_Video/Intersection_1080p.MP4")
    srt: Path | None = None        # auto-derived from the video name if present
    outdir: Path = Path("output")
    name: str = "intersection"
    detect: DetectCfg = field(default_factory=DetectCfg)
    track: TrackCfg = field(default_factory=TrackCfg)
    kinematics: KinematicsCfg = field(default_factory=KinematicsCfg)
    motion: MotionCfg = field(default_factory=MotionCfg)
    post: PostCfg = field(default_factory=PostCfg)
    render: RenderCfg = field(default_factory=RenderCfg)

    # Optional hard exclusion polygons in pixel coords, e.g. a rooftop that the
    # detector hallucinates cars on. Normally UNNECESSARY: static false
    # positives are removed automatically by the stationary filter.
    exclude_rects: list = field(default_factory=list)

    def paths(self):
        o = Path(self.outdir)
        o.mkdir(parents=True, exist_ok=True)
        return {
            "raw":    o / f"{self.name}_detections.parquet",
            "tracks": o / f"{self.name}_tracks_raw.parquet",
            "final":  o / f"{self.name}_l2_tracks.parquet",
            "calib":  o / f"{self.name}_calibration.json",
            "csv":    o / f"{self.name}_tracks.csv",
            "video":  o / f"{self.name}_annotated.mp4",
            "report": o / f"{self.name}_report.json",
        }

    def dump(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2, default=str))
