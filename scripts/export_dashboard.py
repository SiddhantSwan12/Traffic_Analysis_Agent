"""Export real pipeline results for the serverless, read-only dashboard."""
import argparse
import json
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--dest", type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.source / "dashboard" / "backend"))
from store import Store, IDX_MODE

def write(path, value):
    target = args.dest / (path + ".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False), encoding="utf-8")

t0 = time.time()
print("Loading existing analysis...", flush=True)
s = Store()
s.load()
print(f"Analysis loaded in {time.time()-t0:.1f}s", flush=True)
write("health", {"ready": True, "error": None})
write("meta", {**s.meta().__dict__, "modes": IDX_MODE, "n_chunks": s.n_chunks(), "totals": s.insights.get("totals", {})})
write("insights", s.insights)
write("reasoning", s.reasoning)
write("flow", {k: v for k, v in s.flow.items() if k != "time_space"})
for key, value in s.flow.get("time_space", {}).items():
    write("flow/timespace/" + key.replace("->", "--"), value)
for idx in range(s.n_chunks()):
    write(f"chunk/{idx}", s.chunk(idx))
for kind in ("approaches", "desire_lines", "queues", "trajectories"):
    write("map/" + kind, s.geojson(kind))
print("Exporting per-track details...", flush=True)
for display_id in s.moving["display_id"].dropna().unique():
    write(f"track/{int(display_id)}", s.track_detail(int(display_id)))
print(f"Export complete in {time.time()-t0:.1f}s", flush=True)
