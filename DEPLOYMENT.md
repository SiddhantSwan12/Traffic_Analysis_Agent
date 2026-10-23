# Vercel dashboard

The Vercel project uses `dashboard/frontend` as its root directory and sets
`NEXT_PUBLIC_RESULTS_ONLY=1` (also declared in `vercel.json`).
The build downloads a checksummed archive of the real, existing analysis from
the `dashboard-results-v1` GitHub release. Large video and result files remain
outside Git history. Next.js serves the exported JSON and a 720p H.264 copy of
the original footage from the same origin, including native video seeking.

This deployment displays the existing 399-second analysis, including track
selection, overlays, maps, traffic-flow diagrams and findings. Pipeline controls
are hidden because launching Python/GPU jobs requires the original backend.
Without `NEXT_PUBLIC_RESULTS_ONLY=1`, the local Python-backed workflow remains
available.

To regenerate results from a local checkout containing the source video and
`output/intersection_1080p_l2_tracks.parquet`:

```
python scripts/export_dashboard.py --source /path/to/local/project --dest dashboard/frontend/public/data
```

For a new snapshot, package the generated `data/` directory and `video.mp4` in a
tar.gz release asset, then update its release URL and SHA-256 in
`dashboard/frontend/scripts/prepare-results.mjs` before redeploying.
