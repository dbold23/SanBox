"""
STEP 1: Data Prep 4 Hologram Reconstruction and Preprocessing
=====================================================================

What this shi finna hypothetically do
1. Loads raw hologram 
2. Reconstructs focused images at multiple depth planes
3. Automatically finds best focus plane
4. Preprocesses images for ML pipeline
5. Saves standardized images ready for annotation

PSEUDOCODE IMPLEMENTATION
"""

import numpy as np
import cv2
import os
from pathlib import Path
import sys

# Add parent directory to path to import config
sys.path.append(str(Path(__file__).parent.parent))
from config import CONFIG


# ============================================================================
# HOLOGRAM RECONSTRUCTION FUNCTIONS
# ============================================================================

def reconstruct_hologram(raw_hologram, depth_microns):
    """
    Thought is try to reconstruct hologram at given depth

    INPUT:
        raw_hologram = 2D array or 1D tensor if your feeling romatnic (interference pattern from camera)
        depth_microns = reconstruction depth in microns???(idk ur shi)

    OUTPUT:
        reconstructed_image = focused image at specified depth

    NOTE: This is SIMPLIFIED pseudocode. Replace with your actual
          holography reconstruction library idk wtf ur using
    """

    # 1:  parameters
    wavelength = CONFIG['WAVELENGTH']  # microns question MARK
    pixel_size = CONFIG['PIXEL_SIZE']  # also microns not the french president

    #2: Convert to frequency domain
    hologram_fft = np.fft.fft2(raw_hologram)
    hologram_fft_shifted = np.fft.fftshift(hologram_fft)

    # 3: Create frequency coordinate grids
    height, width = raw_hologram.shape
    fx = np.fft.fftfreq(width, d=pixel_size)
    fy = np.fft.fftfreq(height, d=pixel_size)
    FX, FY = np.meshgrid(fx, fy)

    # 4: Angular Spectrum Propagation
    #  implements the diffraction formula ?
    k = 2 * np.pi / wavelength  # Wave number
    propagation_phase = np.exp(1j * k * depth_microns *
                               np.sqrt(1 - (wavelength * FX)**2 - (wavelength * FY)**2))

    # Apply propagation in frequency domain
    propagated_fft = hologram_fft_shifted * propagation_phase

    # 5: Transform back to spatial domain
    propagated_fft_unshifted = np.fft.ifftshift(propagated_fft)
    reconstructed_complex = np.fft.ifft2(propagated_fft_unshifted)

    #6: Extract intensity (just fuckin amplitude squared)
    reconstructed_intensity = np.abs(reconstructed_complex)**2

    return reconstructed_intensity


def reconstruct_hologram_stack(raw_hologram):
    """
    PSEUDOCODE: Reconstruct hologram at multiple depths agian idk what u use

    INPUT: raw_hologram = 2D interference pattern
    OUTPUT: List of images at different depths
    """

    depth_planes = []
    depths = []

    # Reconstruct at each depth
    for depth in range(CONFIG['DEPTH_MIN'],
                      CONFIG['DEPTH_MAX'],
                      CONFIG['DEPTH_STEP']):

        print(f"  Reconstructing at depth {depth} μm...")

        # Reconstruct at this depth
        reconstructed = reconstruct_hologram(raw_hologram, depth)

        depth_planes.append(reconstructed)
        depths.append(depth)

    return depth_planes, depths


def find_best_focus(image_stack):
    """
    PSEUDOCODE: Automatically find bestest focused image

    INPUT: List of images at different depths
    OUTPUT: Index of sharpest image

    METHOD: Uses variance of Laplacian as focus metric
            (higher variance = sharper edges = better focus) (idk if yall have a better way or method)
    """

    focus_scores = []

    for image in image_stack:
        # Normalize image
        normalized = (image - image.min()) / (image.max() - image.min() + 1e-10)
        normalized = (normalized * 255).astype(np.uint8)

        # Calculate Laplacian (edge detection)
        laplacian = cv2.Laplacian(normalized, cv2.CV_64F)

        # Variance of Laplacian = focus metric
        focus_score = laplacian.var()
        focus_scores.append(focus_score)

    # Return index with highest score (sharpest)
    best_index = np.argmax(focus_scores)

    return best_index, focus_scores


# ============================================================================
# IMAGE PREPROCESSING FUNCTIONS
# ============================================================================

def preprocess_image(image):
    """
    PSEUDOCODE: Standardize image for ML pipeline

    INPUT: Raw reconstructed image
    OUTPUT: Preprocessed image (512x512 wit better contrast n stuff)
    """

    # 1: Normalize to 0-255 range
    normalized = (image - image.min()) / (image.max() - image.min() + 1e-10)
    normalized = (normalized * 255).astype(np.uint8)

    #2: Apply CLAHE (look it up if you want)
    #applies local contrast 
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(normalized)

    #3: Resize to standard dimensions
    resized = cv2.resize(enhanced,
                        (CONFIG['IMAGE_WIDTH'], CONFIG['IMAGE_HEIGHT']),
                        interpolation=cv2.INTER_CUBIC)
"""
    # 4: Optional idk what you do for background 
    # Helps separate colonies from background if u think its necessary uncomment it
    blurred_background = cv2.GaussianBlur(resized, (51, 51), 0)
    foreground = cv2.subtract(resized, blurred_background)

    # Clip negative values
    foreground = np.clip(foreground, 0, 255).astype(np.uint8)

    return foreground

"""
# ============================================================================
# MAIN PIPELINE FUNCTION
# ============================================================================

def process_raw_holograms():
    """
    MAIN PSEUDOCODE: Process all raw holograms

    FOR each hologram file what you do:
        1. Load raw hologram
        2. Reconstruct at multiple depths
        3. Find best focus
        4. Preprocess best image
        5. Save for annotation
    """

    raw_dir = Path(CONFIG['RAW_HOLOGRAM_DIR'])
    output_dir = Path(CONFIG['RECONSTRUCTED_DIR'])

    # Find all hologram files
    # Adjust extensions based on your file format agian idk what u use jus keep relevant one but added bunch o shi to look for you(.tif, .png, .npy, etc.)
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
            raw_hologram = cv2.imread(str(hologram_path), cv2.IMREAD_GRAYSCALE)

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

