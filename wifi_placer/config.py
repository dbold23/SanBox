"""
Shared configuration constants for the WiFi placement pipeline.
"""
import numpy as np

# --- Gaussian Splat Loading ---
OPACITY_THRESHOLD = 0.1
SH_C0 = 0.28209479177387814

# --- Voxelization ---
VOXEL_SIZE = 0.05              # meters per voxel edge
DENSITY_THRESHOLD = 0.3        # voxels above this are "occupied"
GAUSSIAN_CUTOFF_SIGMAS = 3.0   # evaluate Gaussian within 3 sigma
MAX_VOXEL_GRID_DIM = 512       # safety cap per axis

# --- Floor Extraction ---
FLOOR_SEARCH_BAND = 0.10       # meters
SLICE_HEIGHT_ABOVE_FLOOR = 1.0 # meters
SLICE_THICKNESS = 0.10         # meters (half-thickness)

# --- WiFi Propagation (COST231 Multi-Wall) ---
FREQUENCIES = {
    "2.4": 2.4e9,
    "5.0": 5.0e9,
}
DEFAULT_TX_POWER_DBM = 20.0
SPEED_OF_LIGHT = 3e8
MIN_RSSI_DBM = -65.0
REFERENCE_WALL_THICKNESS = 0.15  # meters

# Wall attenuation (dB per wall) keyed by density category
WALL_ATTENUATION_DB = {
    "2.4": {"light": 4.0, "medium": 8.0, "heavy": 15.0},
    "5.0": {"light": 6.0, "medium": 12.0, "heavy": 25.0},
}

# --- Optimizer (Simulated Annealing) ---
SA_T_MAX = 1000.0
SA_T_MIN = 1.0
SA_COOLING_RATE = 0.995
SA_STEPS_PER_TEMP = 50
SA_MAX_ROUTERS = 5
SA_COVERAGE_IMPROVEMENT_THRESHOLD = 0.03

# --- Visualization ---
HEATMAP_COLORMAP = "RdYlGn"
ROUTER_MARKER_RADIUS = 0.15
ROUTER_MARKER_COLOR = [1.0, 0.0, 0.0]
