# Benchmark

Measures segmentation **correctness** against hand-labeled ground truth, and
compares pipeline variants (ablations) so you can see what each stage earns.

## 1. Label the frames

`frames/` holds 30 frames sampled across the hard cases — bright rock, sandy
floor, dark water, and vent smoke (from the vent, sand, and V4361 clips).

For each frame, make a binary **ground-truth mask**: **white (255) = seafloor /
ground, black (0) = everything else** (water, smoke, fauna, gear). Save it in
`gt/` with the **same filename** as the frame (e.g. `gt/vent_0638.png`).

Any tool that exports a binary PNG works. Two easy options:

- **FiftyOne** (`pip install fiftyone`) — has SAM-assisted labeling: click the
  seafloor and it fills the mask, then export segmentation PNGs.
- **Label Studio** or **CVAT** — draw the seafloor polygon, export as a mask.

You don't need all 30 to start; even 10–15 spanning the terrains gives a signal.

## 2. Run the benchmark

```bash
python benchmark.py
```

It scores every variant on every labeled frame and prints a table:

| metric | meaning |
|--------|---------|
| IoU / Dice | overlap of predicted vs. true seafloor (higher = better) |
| **Precision** | of what we called ground, how much really is — **catches smoke/water false positives** |
| Recall | of the true ground, how much we caught — catches missed floor |

Results are written to `benchmark/results.csv` (per-variant means) and
`benchmark/results_per_frame.csv`.

## Variants compared (ablations)

- `current (auto, s=0.4)` — the shipped default
- `no texture gate` — shows how much the texture gate buys (expect precision to
  drop on the vent, where smoke leaks back in)
- `fixed texture 1.5` — the pre-"relative threshold" behavior
- `strict (s=0.0)` / `aggressive (s=0.8)` — the sensitivity extremes

Note: this scores **per-frame** segmentation. Temporal stability (flicker) is a
separate axis, measured on full-clip runs via the coverage jitter.
