# Software Infrastructure Plan: In-Situ Lens-Free HAB Imager

This document lays out every piece of software needed to take the current
bench pipeline (reconstruct -> autofocus -> segment) to a deployed,
solar/cellular pier instrument that reports hourly harmful-algal-bloom (HAB)
genus counts to paying users.

It is organized as five layers. Each layer lists its components, the
technology choice, what already exists in this repo, and the build order.
Phases at the end tie the layers to milestones.

```
 ┌─────────────────────────────────────────────────────────────────────────┐
 │ 5. PRODUCT       dashboard · alerts · public API · billing              │
 ├─────────────────────────────────────────────────────────────────────────┤
 │ 4. CLOUD         ingest API · object store · Postgres · job queue       │
 │                  reconstruction workers · model registry · monitoring   │
 ├─────────────────────────────────────────────────────────────────────────┤
 │ 3. ML PLATFORM   labeling tool · dataset versioning · training jobs     │
 │                  evaluation harness · export to edge                    │
 ├─────────────────────────────────────────────────────────────────────────┤
 │ 2. EDGE          acquisition · on-device reconstruction · inference     │
 │                  store-and-forward · OTA · health telemetry             │
 ├─────────────────────────────────────────────────────────────────────────┤
 │ 1. CORE LIBRARY  holography physics · autofocus · preprocessing         │
 │                  segmentation · object extraction · schemas             │
 └─────────────────────────────────────────────────────────────────────────┘
```

---

## Layer 1: Core library (`holo/`)

Pure Python, no hardware or cloud dependencies. Everything above imports
from here, so it must be importable on a Raspberry Pi and on a GPU box.

| Component | Purpose | Status |
|---|---|---|
| `holo/optics.py` | Angular-spectrum propagator, transfer function cache, depth scan | **Done** in `Dataprep.py`; move here |
| `holo/autofocus.py` | Edge-sparsity and Laplacian metrics, coarse-to-fine depth search | **Done** (coarse only); add fine search |
| `holo/preprocess.py` | Normalize, CLAHE, background subtraction, tiling to 512x512 | **Done** in `Dataprep.py` |
| `holo/segment.py` | U-Net inference wrapper, threshold, connected components | Model exists in `Modeltrain.py`; inference wrapper missing |
| `holo/objects.py` | Extract per-object crops, size, aspect ratio, chain length, depth | **Missing** |
| `holo/classify.py` | Genus classifier on crops (Pseudo-nitzschia, dinoflagellate, other, debris) | **Missing** |
| `holo/schema.py` | Pydantic models for `Frame`, `Detection`, `Sample`, `DeviceHealth` | **Missing** |
| `holo/io.py` | Read/write holograms (16-bit TIFF, npy), metadata sidecar JSON | Partial |
| `tests/` | Synthetic hologram round-trip, autofocus, schema validation | **Started** |

Key design rules:

- Per-object depth comes for free from holography. Store it. It is the one
  thing lens-based competitors cannot report and it separates plankton
  from surface film and sensor dust.
- Every function takes and returns numpy arrays plus a metadata dict. No
  global `CONFIG` reads inside library functions; pass parameters
  explicitly so the edge and cloud can use different settings.
- Package with `pyproject.toml`, publish to a private index or install from
  git. Pin numpy and OpenCV versions because the Pi build is fragile.

Build order: refactor existing code into the package (1 day), object
extraction (2 days), inference wrapper (1 day), schemas (half a day).

---

## Layer 2: Edge device (`edge/`)

Runs on a Raspberry Pi 4 or 5 inside the Relay enclosure. Must survive
power loss, no network for days, and a full SD card.

| Component | Purpose | Tech |
|---|---|---|
| `edge/acquire.py` | Trigger laser, capture raw Bayer frame, pump control, timing | `picamera2`, GPIO via `gpiozero` |
| `edge/pipeline.py` | Reconstruct, autofocus, segment, extract objects on-device | Core library, ONNX Runtime |
| `edge/store.py` | SQLite queue of `Sample` records + crops; rotate raw frames by age and disk % | SQLite, `pathlib` |
| `edge/uplink.py` | Store-and-forward over cellular; send summaries first, crops second, raw never unless requested | HTTPS to ingest API, MQTT optional |
| `edge/health.py` | Battery, solar, temperature, laser current, disk, last-successful-upload | Same telemetry format as Relay stations |
| `edge/ota.py` | Pull signed release tarballs, verify, swap, rollback on boot failure | Reuse `Relay-OTA` design |
| `edge/scheduler` | systemd timers: capture every N minutes, upload hourly, health every 5 min | systemd |
| `edge/provision.sh` | Flash image, set device ID and API key, enrol with cloud | Shell + cloud-init style |

Decisions:

- **Reconstruct on device, upload only detections.** A 4056x3040 raw frame
  is 25 MB. Cellular data is the largest recurring cost, so send the
  `Sample` JSON (a few KB) and the object crops (tens of KB). Keep a rolling
  window of raw frames locally so a scientist can request one for QA.
- **Inference runtime is ONNX Runtime**, not TensorFlow. The Keras U-Net
  exports to ONNX once and the Pi runs it without a 500 MB framework.
  Quantize to int8 if the 512x512 pass exceeds a few seconds.
- **Downsample the hologram for autofocus, reconstruct full-res once.**
  Autofocus at 512x512 over 60 depths is fast; full-res reconstruction at
  the chosen depth is one FFT pair.
- Everything under `edge/` is Docker-free. Use a plain venv on Raspberry Pi
  OS Lite so OTA is a tarball swap and the image stays small.

Fleet identity: each device has a UUID, an Ed25519 key for signing uploads,
and a site record (lat/lon, depth below surface, pump flow rate). The cloud
rejects uploads with bad signatures.

---

## Layer 3: ML platform (`ml/`)

Everything needed to go from raw crops to a versioned model the edge can
download.

| Component | Purpose | Tech |
|---|---|---|
| Labeling tool | Grid of crops, keyboard-driven genus labels, per-object depth shown, disagreement queue | Reuse `SharkScarAnnotator` patterns; Label Studio is the fallback |
| Dataset registry | Immutable dataset versions: list of `(crop_id, label, labeler, site, date)` | DVC or plain Parquet in object store, versioned by hash |
| `ml/train_segment.py` | U-Net training as today, plus tile sampling and mixed precision | Keras, existing `Modeltrain.py` |
| `ml/train_classify.py` | Small CNN or fine-tuned EfficientNet-B0 on crops, class weighting | Keras or PyTorch |
| `ml/eval.py` | Per-genus precision/recall, count error vs microscope counts, calibration | scikit-learn, matplotlib |
| `ml/export.py` | Keras -> ONNX, int8 quantization, smoke-test on a Pi image | `tf2onnx`, `onnxruntime` |
| Model registry | `models/<name>/<version>/{model.onnx, metrics.json, dataset_hash}` in object store; edge pulls by tag | S3-style bucket + a `latest` pointer per site class |
| Experiment tracking | Loss curves, hyperparameters, dataset version per run | MLflow (self-hosted) or Weights & Biases free tier |

Validation harness, the one that matters for the pitch:

- Weekly ground-truth file: `validation/<site>/<date>.csv` with microscope
  cells/L per genus from the volunteer program.
- `ml/eval.py --site santacruz_wharf` produces a scatter plot of instrument
  count vs microscope count per genus with R² and bias. This plot is the
  deliverable for Sea Grant, I-Corps, and the first customer.

---

## Layer 4: Cloud backend (`cloud/`)

Small, boring, cheap. One VM plus object storage runs the first ten devices.

| Component | Purpose | Tech |
|---|---|---|
| Ingest API | `POST /v1/samples`, `POST /v1/crops`, `POST /v1/health`, signature check, idempotent by sample ID | FastAPI + Uvicorn |
| Object store | Crops, requested raw frames, datasets, models | Cloudflare R2 or Backblaze B2 (no egress fees) |
| Database | Devices, sites, samples, detections, users, alerts | Postgres, TimescaleDB extension for the time series |
| Job queue | Re-reconstruction of requested raw frames, batch re-classification after a model update, nightly aggregates | Redis + RQ, or Postgres `SKIP LOCKED` queue |
| Aggregation | Hourly and daily cells/L per genus per site; rolling anomaly score | SQL views + a cron worker |
| Auth | API keys for devices, magic-link login for users, org membership | FastAPI Users or Clerk |
| Monitoring | Device last-seen, upload lag, battery trend; page on silence > 6 h | Grafana + Prometheus, or Healthchecks.io for the first version |
| Backups | Nightly `pg_dump` to object store, weekly restore test | cron |
| Infra as code | One `docker-compose.yml` for dev, Terraform or a single Ansible playbook for the VM | Docker, Ansible |

Data model (minimum):

```
devices(id, site_id, pubkey, firmware_version, last_seen_at)
sites(id, name, lat, lon, depth_m, partner_org)
samples(id, device_id, captured_at, volume_ml, focus_depth_um, n_objects, model_version)
detections(id, sample_id, genus, confidence, depth_um, major_axis_um, chain_length, crop_key)
health(device_id, ts, battery_v, solar_w, temp_c, disk_pct, laser_ma)
ground_truth(site_id, date, genus, cells_per_l, source)
alerts(id, site_id, genus, threshold, triggered_at, acknowledged_by)
```

Retention: detections forever, crops 2 years, raw frames 30 days unless
flagged, health 1 year at full resolution then downsampled.

---

## Layer 5: Product surface (`web/`)

| Component | Purpose | Tech |
|---|---|---|
| Site dashboard | Time series per genus, latest crops, device health, comparison overlay with microscope counts | Next.js or SvelteKit, Plotly or ECharts |
| Alerting | Threshold per genus per site; email + SMS; escalation if unacknowledged | Postmark + Twilio |
| Public API | Read-only `GET /v1/sites/{id}/counts?from=&to=` with API keys, rate limits, OpenAPI docs | FastAPI, same service as ingest |
| Data export | CSV and NetCDF (CF-compliant, so NOAA and ERDDAP can ingest it) | `xarray` |
| Public status page | One page per partner site showing last 7 days; this is the marketing | Static build from the API |
| Billing | Per-device subscription; invoice for grants and agencies | Stripe, manual invoices to start |

The first customer does not need billing. They need the alert SMS to arrive
before the closure notice would have.

---

## Cross-cutting

**Repository layout** (single monorepo until there is a second engineer):

```
holo-imager/
  holo/            core library
  edge/            device runtime
  ml/              training, eval, export
  cloud/           API, workers, migrations
  web/             dashboard
  hardware/        BOM, enclosure, wiring, calibration procedure
  docs/            this file, runbooks, calibration, deployment checklist
  tests/
  pyproject.toml
  docker-compose.yml
  .github/workflows/
```

**CI** (GitHub Actions):

- `test.yml`: pytest for `holo/` and `cloud/` on every push.
- `edge-build.yml`: build the Pi tarball on an arm64 runner or with QEMU,
  run the synthetic-hologram test inside it, attach to a release.
- `model-eval.yml`: on a new model tag, run `ml/eval.py` against the
  frozen validation set and refuse to publish if any genus recall drops.

**Calibration and QA** (software side of hardware):

- `hardware/calibrate.py`: image a USAF target and a bead standard, fit
  effective pixel size and source-to-sensor distance, write `optics.json`
  to the device.
- Weekly automatic blank: pump filtered seawater, confirm object count near
  zero, flag fouling if the background drifts.

**Security**: signed uploads, no inbound ports on the device, API keys
scoped per device, secrets in environment variables only, dependabot on.

**Documentation**: a runbook per failure mode (device silent, counts jump,
pump stall, fouling), a deployment checklist, and the calibration
procedure. Write them while doing the thing the first time.

---

## Phases

| Phase | Weeks | Deliverable | Layers |
|---|---|---|---|
| 0. Bench | 1-2 | Real hologram of wharf water reconstructed and autofocused with this repo | 1 |
| 1. Package | 3 | `holo/` package with tests, object extraction, ONNX export of U-Net | 1, 3 |
| 2. Labels | 4-6 | 500+ labeled crops across 4 classes, first classifier, eval report | 3 |
| 3. Edge | 7-9 | Pi captures, reconstructs, classifies, queues, uploads over cellular | 2, 4 |
| 4. Cloud | 9-11 | Ingest API, Postgres, object store, health monitoring, one dashboard page | 4, 5 |
| 5. Validate | 12-16 | Four weeks on a pier next to a volunteer microscope count; scatter plot | 3, 5 |
| 6. Alert | 17-18 | Threshold alerts by SMS to one partner; public status page | 5 |

Phase 5 is the milestone that unlocks grants and the first customer.
Everything before it should be the minimum needed to reach it.

## What to buy or sign up for now

- Raspberry Pi 5, HQ camera module (remove lens), 650 nm laser diode module
  with driver, small peristaltic pump, flow cell or two coverslips with a
  spacer. Under 300 USD.
- Cloudflare R2 bucket, one small VPS (Hetzner or similar), a domain.
- Cellular data SIM already used by the Relay stations.
- Nothing else until Phase 5.
