from pathlib import Path
from .constants import PARCELS_DIR

def find_parcels(directory: Path = PARCELS_DIR) -> Path | None:
    """Return most recent GeoJSON or GPKG from directory, if any."""
    if not directory.exists():
        return None
    candidates = list(directory.glob("*.geojson")) + list(directory.glob("*.gpkg"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)
