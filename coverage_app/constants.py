from pathlib import Path

# Project root directory
ROOT = Path(__file__).resolve().parents[1]

# Static files location
STATIC_DIR = ROOT / "static"
MAPS_DIR = STATIC_DIR / "maps"

# Directory to search for parcel files
PARCELS_DIR = ROOT / "parcels"
