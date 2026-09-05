# SanBox: Lens-Free Holographic Microscopy Pipeline

Reconstructs in-line digital holograms with the angular spectrum method,
auto-focuses with an edge-sparsity criterion, preprocesses the focused
image, and trains a U-Net to segment objects (bacterial colonies today,
plankton next).

See `docs/INFRASTRUCTURE.md` for the full plan to turn this into a
deployed in-situ harmful-algal-bloom imager.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

TensorFlow is only needed for `Modeltrain.py`; the reconstruction pipeline
runs with numpy and OpenCV alone.

## Configure

All settings live in `config.py`. Override any of them with an environment
variable prefixed `HOLO_`, for example:

```bash
HOLO_WAVELENGTH=0.532 HOLO_PIXEL_SIZE=1.12 HOLO_DEPTH_MIN=-2000 python Dataprep.py --raw
```

Defaults assume a 650 nm laser diode and a Raspberry Pi HQ sensor with the
lens removed. Depths are signed: the object side of the sensor is negative,
so the default scan runs from -3000 to -200 microns.

## Input holograms

Save holograms exactly as the sensor recorded them: raw 8-bit or 16-bit
grayscale TIFF or PNG, black-level pedestal included. Do not auto-contrast
or min-max stretch them. Stretching rescales the interference fringes
relative to the background term and shifts the autofocus by hundreds of
microns on synthetic tests, while raw, gain-scaled, and pedestal-offset
inputs all focus correctly.

## Run

```bash
# Reconstruct every hologram in data/raw_holograms/ -> data/reconstructed/
python Dataprep.py --raw

# Only preprocess images that are already reconstructed
python Dataprep.py --existing

# Train the U-Net on data/train/{images,masks} and data/validation/{images,masks}
python Modeltrain.py
```

## Test

```bash
pytest tests
```

The tests synthesize a hologram from a known object plane, propagate it a
known distance, and confirm that reconstruction and autofocus recover that
distance with no NaN values.
