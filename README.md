# MBARI ROV Seafloor Segmentation

Segment the seafloor ("ground") in MBARI remotely-operated-vehicle (ROV) dive
video using Segment Anything (SAM 2.1). Given a clip, it produces a green
overlay video, per-frame binary masks, and a CSV of how much of each frame the
ground covers.

## Sample footage

`data/` contains three 1-second clips from MBARI ROV dive **V4361**
(2021-10-06), taken from the public
[`mbari-org/deepsea-ai`](https://github.com/mbari-org/deepsea-ai) test fixtures.
Drop your own `.mp4` clips into `data/` to run on other footage.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Everything runs on CPU. The SAM 2.1 tiny weights (`sam2.1_t.pt`, ~150 MB)
download automatically from the ultralytics GitHub release on first run.

## Run

```bash
python segment_ground.py --video data/V4361_20211006T162656Z_h265_1sec.mp4
```

Useful flags:

| flag | default | meaning |
|------|---------|---------|
| `--model` | `sam2.1_t.pt` | SAM weights (`sam2.1_s.pt`, `FastSAM-s.pt`, ... also work) |
| `--stride` | `1` | process every Nth frame to save CPU time |
| `--imgsz` | `640` | inference resolution |
| `--max-frames` | `0` | cap processed frames (0 = all) |
| `--out` | `outputs` | output directory |

## How "ground" is found

ROV footage looks down or forward at the bottom, so the seafloor occupies the
lower part of the frame and is continuous with the bottom edge. The script:

1. Prompts SAM with positive points spread across the lower ~45% of the frame
   and negative points near the top (open water).
2. Keeps only mask components connected to the bottom edge, dropping midwater
   detections (fish, marine snow, lit particles).
3. Lightly smooths the mask across frames to reduce flicker.

## Outputs (`outputs/`)

- `overlay.mp4` — original footage with the ground tinted green and outlined.
- `masks/000123.png` — per-frame binary ground masks.
- `coverage.csv` — ground coverage fraction per processed frame.
