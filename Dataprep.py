"""
STEP 1: Data prep for hologram reconstruction and preprocessing
=====================================================================

What this pipeline does
1. Loads raw hologram 
2. Reconstructs focused images at multiple depth planes
3. Automatically finds best focus plane
4. Preprocesses images for ML pipeline
5. Saves standardized images ready for annotation

Run:  python Dataprep.py --raw        (reconstruct raw holograms)
      python Dataprep.py --existing   (only preprocess existing images)
"""

import numpy as np
import cv2
import os
from pathlib import Path
import sys

# config.py lives next to this file
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CONFIG, ensure_dirs


# ============================================================================
# HOLOGRAM RECONSTRUCTION FUNCTIONS
# ============================================================================

def angular_spectrum_transfer(height, width, depth_microns,
                              wavelength=None, pixel_size=None):
    """
    Build the angular-spectrum transfer function H(fx, fy; z).

    Returns a complex array laid out in the SAME (unshifted) frequency order
    that np.fft.fft2 produces, so it can be multiplied directly against
    fft2(hologram) with no fftshift on either side.

    Evanescent frequencies, where (lambda*fx)^2 + (lambda*fy)^2 > 1, are
    zeroed. Propagating them would take the square root of a negative
    number and fill the reconstruction with NaN.
    """
    wavelength = CONFIG['WAVELENGTH'] if wavelength is None else wavelength
    pixel_size = CONFIG['PIXEL_SIZE'] if pixel_size is None else pixel_size

    fx = np.fft.fftfreq(width, d=pixel_size)   # cycles per micron
    fy = np.fft.fftfreq(height, d=pixel_size)
    FX, FY = np.meshgrid(fx, fy)

    k = 2 * np.pi / wavelength
    argument = 1.0 - (wavelength * FX) ** 2 - (wavelength * FY) ** 2
    propagating = argument > 0

    kz = np.zeros_like(argument)
    kz[propagating] = k * np.sqrt(argument[propagating])

    transfer = np.exp(1j * kz * depth_microns)
    transfer[~propagating] = 0.0
    return transfer


def reconstruct_hologram(raw_hologram, depth_microns, return_complex=False):
    """
    Reconstruct an in-line hologram at a given depth with the angular
    spectrum method.

    INPUT:
        raw_hologram   = 2D intensity pattern from the sensor
        depth_microns  = signed propagation distance in microns. See the
                         DEPTH_* comment in config.py for the sign convention.
        return_complex = return the complex field instead of intensity

    OUTPUT:
        reconstructed intensity (or complex field) at that depth
    """
    field = np.asarray(raw_hologram, dtype=np.float64)

    # Dividing out the mean removes the DC term so the object, not the
    # reference beam, dominates the reconstruction.
    if CONFIG['NORMALIZE_HOLOGRAM']:
        mean = field.mean()
        if mean > 0:
            field = field / mean

    height, width = field.shape
    spectrum = np.fft.fft2(field)
    transfer = angular_spectrum_transfer(height, width, depth_microns)
    reconstructed_complex = np.fft.ifft2(spectrum * transfer)

    if return_complex:
        return reconstructed_complex
    return np.abs(reconstructed_complex) ** 2


def depth_scan_values():
    """Depths to reconstruct at, taken from CONFIG. Supports float steps."""
    depths = np.arange(CONFIG['DEPTH_MIN'],
                       CONFIG['DEPTH_MAX'] + 1e-9,
                       CONFIG['DEPTH_STEP'])
    if len(depths) == 0:
        raise ValueError("DEPTH_MIN/DEPTH_MAX/DEPTH_STEP produce an empty scan")
    return depths


def reconstruct_hologram_stack(raw_hologram, verbose=True):
    """
    Reconstruct a hologram at every depth in the configured scan.

    INPUT:  raw_hologram = 2D interference pattern
    OUTPUT: (list of intensity images, list of depths)

    The FFT of the hologram is computed once and reused for every depth.
    """
    field = np.asarray(raw_hologram, dtype=np.float64)
    if CONFIG['NORMALIZE_HOLOGRAM']:
        mean = field.mean()
        if mean > 0:
            field = field / mean

    height, width = field.shape
    spectrum = np.fft.fft2(field)

    depth_planes = []
    depths = []
    for depth in depth_scan_values():
        if verbose:
            print(f"  Reconstructing at depth {depth:.1f} um...")
        transfer = angular_spectrum_transfer(height, width, depth)
        reconstructed = np.abs(np.fft.ifft2(spectrum * transfer)) ** 2
        depth_planes.append(reconstructed)
        depths.append(float(depth))

    return depth_planes, depths


def _gradient_magnitude(image):
    gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return np.hypot(gx, gy)


def _normalize_scores(values):
    values = np.asarray(values, dtype=np.float64)
    span = values.max() - values.min()
    return (values - values.min()) / (span + 1e-12)


def focus_score(image, method="laplacian"):
    """
    Single-image sharpness score. Higher is sharper.

    These per-image metrics are provided for completeness but are NOT
    reliable on in-line holograms: defocus fringes are themselves sharp,
    high-frequency structure, so both metrics routinely peak at the wrong
    depth. Use find_best_focus(method="dark_edge") for real work.

    method="laplacian"  variance of the Laplacian (the classic photo metric)
    method="sparsity"   L2/L1 ratio of the gradient magnitude
    """
    image = np.asarray(image, dtype=np.float64)
    image = (image - image.min()) / (image.max() - image.min() + 1e-10)

    if method == "laplacian":
        as_uint8 = (image * 255).astype(np.uint8)
        return float(cv2.Laplacian(as_uint8, cv2.CV_64F).var())

    if method == "sparsity":
        magnitude = _gradient_magnitude(image)
        return float(np.sqrt((magnitude ** 2).sum()) / (magnitude.sum() + 1e-12))

    raise ValueError(f"unknown focus method: {method}")


def dark_edge_components(amplitude, window=9, crop_radius=24):
    """
    Two per-plane measurements used by the "dark_edge" autofocus.

    darkness = -min of the box-filtered amplitude. A solid absorbing object
               in focus is a uniformly dark region; thin defocus fringes
               are removed by the box filter and do not count.
    edge     = strongest gradient inside a crop around that darkest region.
               In focus the object boundary is a single sharp step; out of
               focus it is a soft, ringed shadow.

    Darkness alone ties when a large object's geometric shadow is still
    dark near the sensor plane. Edge alone ties on fringe ridges. Their
    product across the stack does not, on synthetic tests.
    """
    amplitude = np.asarray(amplitude, dtype=np.float64)
    blurred = cv2.blur(amplitude, (window, window))
    cy, cx = np.unravel_index(int(np.argmin(blurred)), blurred.shape)
    y0, x0 = max(cy - crop_radius, 0), max(cx - crop_radius, 0)
    crop = amplitude[y0:cy + crop_radius, x0:cx + crop_radius]
    return -float(blurred.min()), float(_gradient_magnitude(crop).max())


def find_best_focus(image_stack, method="dark_edge"):
    """
    Pick the best-focused plane from a reconstructed depth stack.

    INPUT:  list of intensity images at different depths
    OUTPUT: (index of sharpest image, list of per-plane focus scores)

    method="dark_edge"  (default) product of stack-normalized darkness and
                        edge scores from dark_edge_components(). Chosen
                        because it recovered the true depth in 36/36
                        synthetic trials where Laplacian variance, gradient
                        sparsity, Gini edge sparsity, and the Dubois
                        integrated-amplitude criterion all failed on
                        defocus rings or geometric shadows.
    method="laplacian" / "sparsity"  per-image metrics, see focus_score().

    Scores are always "higher is sharper".
    """
    if method == "dark_edge":
        amplitudes = [np.sqrt(np.clip(np.asarray(img, dtype=np.float64), 0, None))
                      for img in image_stack]
        darkness, edge = zip(*(dark_edge_components(a) for a in amplitudes))
        scores = (_normalize_scores(darkness) * _normalize_scores(edge)).tolist()
    else:
        scores = [focus_score(image, method) for image in image_stack]

    best_index = int(np.argmax(scores))
    return best_index, scores


# ============================================================================
# IMAGE PREPROCESSING FUNCTIONS
# ============================================================================

def preprocess_image(image, subtract_background=False):
    """
    Standardize an image for the ML pipeline.

    INPUT:  Raw reconstructed image
    OUTPUT: Preprocessed image (512x512, contrast-enhanced)

    Pass subtract_background=True to remove the low-frequency background,
    which helps separate colonies when the illumination is uneven.
    """

    # 1: Normalize to 0-255 range
    normalized = (image - image.min()) / (image.max() - image.min() + 1e-10)
    normalized = (normalized * 255).astype(np.uint8)

    # 2: Apply CLAHE for local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized)

    # 3: Resize to standard dimensions
    resized = cv2.resize(enhanced,
                        (CONFIG['IMAGE_WIDTH'], CONFIG['IMAGE_HEIGHT']),
                        interpolation=cv2.INTER_CUBIC)

    # 4: Optional background subtraction
    if subtract_background:
        blurred_background = cv2.GaussianBlur(resized, (51, 51), 0)
        foreground = cv2.subtract(resized, blurred_background)
        # Clip negative values
        return np.clip(foreground, 0, 255).astype(np.uint8)

    return resized


# ============================================================================
# MAIN PIPELINE FUNCTION
# ============================================================================

def process_raw_holograms():
    """
    Process every raw hologram in RAW_HOLOGRAM_DIR.

    Save holograms as the sensor recorded them (raw 8-bit or 16-bit, a
    black-level pedestal is fine). Do NOT auto-contrast or min-max stretch
    them first: that rescales the interference term relative to the DC
    term and throws the autofocus off by hundreds of microns.

    FOR each hologram file:
        1. Load raw hologram
        2. Reconstruct at multiple depths
        3. Find best focus
        4. Preprocess best image
        5. Save for annotation
    """

    raw_dir = Path(CONFIG['RAW_HOLOGRAM_DIR'])
    output_dir = Path(CONFIG['RECONSTRUCTED_DIR'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all hologram files
    # Adjust extensions to match your file format; keep only the relevant
    # entries (.tif, .png, .npy, etc.)
    hologram_files = list(raw_dir.glob('*.tif')) + \
                    list(raw_dir.glob('*.tiff')) + \
                    list(raw_dir.glob('*.png'))

    if len(hologram_files) == 0:
        print(f"WARNING: No hologram files found in {raw_dir}")
        print("Please place your raw hologram files in data/raw_holograms/")
        return

    print(f"Found {len(hologram_files)} hologram files to process\n")

    for i, hologram_path in enumerate(hologram_files, 1):
        print(f"[{i}/{len(hologram_files)}] Processing: {hologram_path.name}")

        try:
            #Load raw hologram
            # ANYDEPTH keeps 16-bit TIFFs at full precision instead of
            # silently downconverting them to 8-bit.
            raw_hologram = cv2.imread(str(hologram_path),
                                      cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)

            if raw_hologram is None:
                print(f"  ERROR: Could not load {hologram_path}")
                continue

            # Reconstruct at multiple depths
            print("  Reconstructing hologram stack...")
            image_stack, depths = reconstruct_hologram_stack(raw_hologram)

            # Find best focus
            print("  Finding best focus plane...")
            best_idx, focus_scores = find_best_focus(image_stack)
            best_depth = depths[best_idx]
            best_image = image_stack[best_idx]

            print(f"  Best focus at depth: {best_depth} μm")

            # Preprocess
            print("  Preprocessing image...")
            processed = preprocess_image(best_image)

            # Save
            output_filename = hologram_path.stem + "_reconstructed.png"
            output_path = output_dir / output_filename
            cv2.imwrite(str(output_path), processed)

            print(f"  Saved: {output_filename}\n")

        except Exception as e:
            print(f"  ERROR processing {hologram_path.name}: {e}\n")
            continue

 


# ============================================================================
# If you already have reconstructed images
# ============================================================================

def process_existing_reconstructed_images():
    """
    PSEUDOCODE: If you already have reconstructed images,
                just preprocess them
    """

    # Put your existing reconstructed images in data/reconstructed/
    reconstructed_dir = Path(CONFIG['RECONSTRUCTED_DIR'])
    output_dir = reconstructed_dir  # Save in same location

    image_files = list(reconstructed_dir.glob('*.tif')) + \
                 list(reconstructed_dir.glob('*.png'))

    for image_path in image_files:
        # Skip already processed files
        if '_processed' in image_path.name:
            continue

        print(f"Processing: {image_path.name}")

        # Load image
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        # Preprocess
        processed = preprocess_image(image)

        # Save
        output_name = image_path.stem + "_processed.png"
        output_path = output_dir / output_name
        cv2.imwrite(str(output_path), processed)

        print(f"  Saved: {output_name}")


# ============================================================================
# MAIN
# ============================================================================

def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Reconstruct raw holograms or preprocess existing images.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--raw", action="store_true",
                       help="reconstruct every hologram in RAW_HOLOGRAM_DIR")
    group.add_argument("--existing", action="store_true",
                       help="preprocess already-reconstructed images in RECONSTRUCTED_DIR")
    args = parser.parse_args(argv)

    ensure_dirs()
    if args.raw:
        process_raw_holograms()
    else:
        process_existing_reconstructed_images()


if __name__ == "__main__":
    main()
