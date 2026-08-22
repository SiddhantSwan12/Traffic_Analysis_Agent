"""Batched tiled YOLO detection for small aerial objects."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import numpy as np


def resolve_weights(model: str) -> str:
    """Accept a local .pt path or an HF repo id; return a local .pt path."""
    p = Path(model)
    if p.suffix == ".pt" and p.exists():
        return str(p)
    from huggingface_hub import list_repo_files, hf_hub_download
    files = [f for f in list_repo_files(model) if f.endswith(".pt")]
    if not files:
        raise FileNotFoundError(f"no .pt weights in HF repo '{model}'")
    # Prefer best.pt, then the shortest name (avoids last.pt / epoch*.pt).
    files.sort(key=lambda f: (f != "best.pt", len(f)))
    return hf_hub_download(model, files[0])


def make_tiles(W: int, H: int, tile: int, overlap: int, max_object_px: int):
    """Tile the frame, and precompute each tile's exclusive 'core' region.

    A detection is kept only from the tile whose core contains its centre. The
    cores partition the frame exactly, so cross-tile duplicates are impossible
    by construction -- no NMS heuristics involved. This is valid as long as the
    overlap exceeds the largest object, which is asserted here.
    """
    if overlap <= max_object_px:
        raise ValueError(
            f"tile_overlap ({overlap}) must exceed max_object_px ({max_object_px}); "
            "otherwise objects can be truncated in every tile that sees them.")

    def starts(total):
        if total <= tile:
            return [0]
        step = tile - overlap
        n = int(np.ceil((total - tile) / step)) + 1
        s = [min(int(round(i * (total - tile) / (n - 1))), total - tile) for i in range(n)]
        return sorted(set(s))

    xs, ys = starts(W), starts(H)

    def cores(s, total):
        """Midpoints of consecutive overlaps become the core boundaries."""
        bounds = [0]
        for a, b in zip(s, s[1:]):
            bounds.append((b + a + tile) // 2)   # midpoint of [b, a+tile]
        bounds.append(total)
        return list(zip(bounds, bounds[1:]))

    cx, cy = cores(xs, W), cores(ys, H)
    tiles = []
    for j, y0 in enumerate(ys):
        for i, x0 in enumerate(xs):
            tiles.append({
                "x0": x0, "y0": y0,
                "x1": min(x0 + tile, W), "y1": min(y0 + tile, H),
                "core": (cx[i][0], cy[j][0], cx[i][1], cy[j][1]),
            })
    return tiles


class TiledDetector:
    def __init__(self, cfg, frame_w: int, frame_h: int):
        import torch
        from ultralytics import YOLO
        self.cfg = cfg
        weights = resolve_weights(cfg.model)
        self.model = YOLO(weights)
        self.device = cfg.device if torch.cuda.is_available() else "cpu"
        self.half = bool(cfg.half) and self.device.startswith("cuda")
        self.model.to(self.device)
        self.names = dict(self.model.names)

        # Tile settings are expressed for 1080p. At 4K an object is twice as
        # many pixels across, so a fixed 200 px overlap would be smaller than
        # the objects it has to contain. Scale the whole tiling with the frame
        # so any resolution works; 1080p input leaves these untouched.
        s = frame_w / 1920.0
        if s > 1.25:
            k = int(round(s))
            cfg = replace(cfg, tile=cfg.tile * k,
                          tile_overlap=cfg.tile_overlap * k,
                          max_object_px=cfg.max_object_px * k)
            self.cfg = cfg
            print(f"  {frame_w}x{frame_h}: scaling tiles x{k} "
                  f"(tile {cfg.tile}px, overlap {cfg.tile_overlap}px). "
                  "A 1080p export runs far faster at equivalent quality.")
        # ultralytics renamed `half` to `quantize`; support whichever this
        # version accepts so fp16 is actually used rather than silently dropped.
        import inspect
        sig = inspect.signature(self.model.predict).parameters
        if not self.half:
            self._prec = {}
        elif "quantize" in sig:
            self._prec = {"quantize": "fp16"}   # 8.4+ spelling
        else:
            self._prec = {"half": True}         # older releases
        self.tiles = make_tiles(frame_w, frame_h, cfg.tile, cfg.tile_overlap,
                                cfg.max_object_px)
        self.W, self.H = frame_w, frame_h

    def warmup(self, frame):
        self(frame)

    def __call__(self, frame: np.ndarray):
        import torch
        """frame BGR -> (boxes Nx4 float32 xyxy, scores N, class_ids N)."""
        crops = [frame[t["y0"]:t["y1"], t["x0"]:t["x1"]] for t in self.tiles]
        res = self.model.predict(
            crops, imgsz=self.cfg.tile, conf=self.cfg.conf, iou=self.cfg.nms_iou,
            device=self.device, verbose=False, augment=False, **self._prec,
        )
        boxes, scores, clss = [], [], []
        for t, r in zip(self.tiles, res):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            b = r.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
            b[:, [0, 2]] += t["x0"]
            b[:, [1, 3]] += t["y0"]
            cxs = 0.5 * (b[:, 0] + b[:, 2])
            cys = 0.5 * (b[:, 1] + b[:, 3])
            kx0, ky0, kx1, ky1 = t["core"]
            keep = (cxs >= kx0) & (cxs < kx1) & (cys >= ky0) & (cys < ky1)
            if not keep.any():
                continue
            boxes.append(b[keep])
            scores.append(r.boxes.conf.detach().cpu().numpy().astype(np.float32)[keep])
            clss.append(r.boxes.cls.detach().cpu().numpy().astype(np.int32)[keep])

        if not boxes:
            return (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32),
                    np.zeros((0,), np.int32))

        boxes = np.concatenate(boxes)
        scores = np.concatenate(scores)
        clss = np.concatenate(clss)
        np.clip(boxes[:, [0, 2]], 0, self.W - 1, out=boxes[:, [0, 2]])
        np.clip(boxes[:, [1, 3]], 0, self.H - 1, out=boxes[:, [1, 3]])

        # Light safety NMS. Core partitioning already removes tile duplicates;
        # this only catches a detector firing twice on one object.
        from torchvision.ops import nms
        k = nms(torch.from_numpy(boxes), torch.from_numpy(scores), 0.75).numpy()
        return boxes[k], scores[k], clss[k]
