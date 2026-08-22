"""DJI .srt flight-telemetry parsing.

Entries are stored positionally: index i == video frame i. The SRT's own
FrameCnt field is not trusted to align with OpenCV's frame numbering.
"""
from __future__ import annotations
from pathlib import Path
import re

_FIELD = {
    "lat":      re.compile(r"\[latitude:\s*([-\d.]+)\]"),
    "lon":      re.compile(r"\[long(?:itude)?:\s*([-\d.]+)\]"),
    "rel_alt":  re.compile(r"\[rel_alt:\s*([-\d.]+)"),
    "abs_alt":  re.compile(r"abs_alt:\s*([-\d.]+)\]"),
    "gb_yaw":   re.compile(r"\[gb_yaw\s*:\s*([-\d.]+)"),
    "gb_pitch": re.compile(r"gb_pitch\s*:\s*([-\d.]+)"),
    "gb_roll":  re.compile(r"gb_roll\s*:\s*([-\d.]+)"),
}
_TS = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[.,]\d+)")


def parse_srt(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    text = path.read_text(errors="ignore")
    blocks = re.split(r"\n\s*\n", text.strip())
    out = []
    for blk in blocks:
        if not blk.strip():
            continue
        rec = {}
        m = _TS.search(blk)
        rec["timestamp"] = m.group(1).replace(",", ".") if m else None
        for k, rx in _FIELD.items():
            m = rx.search(blk)
            rec[k] = float(m.group(1)) if m else None
        out.append(rec)
    return out


def align(entries: list[dict], n_frames: int) -> list[dict]:
    """Pad/trim to exactly n_frames so telemetry[i] is always valid."""
    if not entries:
        return [{}] * n_frames
    if len(entries) >= n_frames:
        return entries[:n_frames]
    return entries + [entries[-1]] * (n_frames - len(entries))


def find_srt(video_path) -> Path | None:
    p = Path(video_path)
    for cand in (p.with_suffix(".srt"), p.with_suffix(".SRT")):
        if cand.exists():
            return cand
    return None
