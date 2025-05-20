"""
radio_viewshed_server.py  ·  Copernicus DSM 10 m
POST /radio_viewshed → GeoJSON MultiPolygon с видимыми пятнами
"""

from __future__ import annotations
import hashlib, math, os, pathlib, time, logging, threading, concurrent.futures
from typing import Dict, Tuple, List

import numpy as np, rasterio, requests
from rasterio.enums import Resampling
from rasterio.features import shapes
from shapely.geometry import MultiPolygon, shape, mapping
from shapely.ops import unary_union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pyproj import Geod, Transformer
from tqdm import tqdm
from numba import njit

# ──────────────────────── logging ────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL,
                    format="%(asctime)s [%(levelname)7s] %(name)s: %(message)s")
log = logging.getLogger("viewshed")

# ──────────────────────── constants ──────────────────────
EARTH_RADIUS_M = 6_371_000.0
geod = Geod(ellps="WGS84")

COP_DIR = pathlib.Path(os.getenv("COP_DIR", "./copernicus")).expanduser()
COP_DIR.mkdir(parents=True, exist_ok=True)
BUCKET = "https://copernicus-dem-30m.s3.amazonaws.com"

# ──────────────────────── DEM loader ─────────────────────
def _key(lat: int, lon: int) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    stem = f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"
    return f"{stem}/{stem}.tif"

def _path(lat: int, lon: int) -> pathlib.Path:
    return COP_DIR / pathlib.Path(_key(lat, lon)).name

def _md5(fname: pathlib.Path, buf: int = 1 << 20) -> str:
    h = hashlib.md5()
    with fname.open("rb") as f:
        chunk = f.read(buf)
        while chunk:
            h.update(chunk)
            chunk = f.read(buf)
    return h.hexdigest()

def _etag(k: str) -> str:
    r = requests.head(f"{BUCKET}/{k}", timeout=20)
    r.raise_for_status()
    return r.headers["ETag"].strip('"')

def _download(lat: int, lon: int) -> pathlib.Path:
    k, dst = _key(lat, lon), _path(lat, lon)
    et = _etag(k)
    if dst.exists() and ("-" in et or _md5(dst) == et):
        log.debug("DEM cache hit %s", dst.name)
        return dst

    log.info("DEM downloading %s …", dst.name)
    url = f"{BUCKET}/{k}"
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    total = int(r.headers.get("Content-Length", 0))
    bar = tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024,
               desc=f"↓ {dst.name}", leave=False)
    tmp = dst.with_suffix(".part")
    with tmp.open("wb") as f:
        for chunk in r.iter_content(1 << 15):
            f.write(chunk)
            bar.update(len(chunk))
    bar.close()

    if "-" not in et and _md5(tmp) != et:
        tmp.unlink()
        raise IOError("MD5 mismatch after download")
    tmp.rename(dst)
    log.info("DEM saved %s (%.1f MiB)", dst.name, total / 1048576)
    return dst

class Tile:
    __slots__ = ("arr", "inv")
    def __init__(self, lat: int, lon: int):
        fp = _download(lat, lon)
        with rasterio.open(fp) as ds:
            self.arr = ds.read(1).astype(np.float32)
            self.arr[self.arr == ds.nodata] = 0
            self.inv = ~ds.transform
        log.debug("Tile %s loaded into RAM", fp.name)

    def sample(self, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
        col, row = self.inv * (lons, lats)
        col = np.clip(col.astype(int), 0, self.arr.shape[1] - 1)
        row = np.clip(row.astype(int), 0, self.arr.shape[0] - 1)
        return self.arr[row, col]

DEM_CACHE: Dict[Tuple[int, int], Tile] = {}
LOCK = threading.Lock()

def tile(lon: float, lat: float) -> Tile:
    key = (int(math.floor(lat)), int(math.floor(lon)))
    with LOCK:
        if key not in DEM_CACHE:
            DEM_CACHE[key] = Tile(*key)
        return DEM_CACHE[key]

def elev_vec(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    return tile(lons[0], lats[0]).sample(lons, lats)

# ──────────────────────── Numba LOS ──────────────────────
@njit(fastmath=True, cache=True)
def _line_of_sight(elev_line, obs_h, obj_h, r_eff, step):
    vis = np.zeros(elev_line.size, np.bool_)
    max_t = -1e20
    for i in range(elev_line.size):
        s = (i + 1) * step
        tan = (elev_line[i] + obj_h - (s * s) / (2 * r_eff) - obs_h) / s
        if tan > max_t:
            max_t = tan
            vis[i] = True
    return vis

# ──────────────────────── FastAPI models ─────────────────
class ViewshedReq(BaseModel):
    lat: float
    lon: float
    height_m: float
    obj_height_m: float = 0
    max_distance_km: float = 100
    grid_step_m: float = 100
    k_factor: float = Field(4/3)

class ViewshedResp(BaseModel):
    geojson: dict
    visible_ratio: float

app = FastAPI(title="Copernicus viewshed", version="1.0")
@app.get("/health")
async def health(): return {"ok": True}

# ──────────────────────── main algorithm ─────────────────
@app.post("/radio_viewshed", response_model=ViewshedResp)
async def radio_viewshed(req: ViewshedReq):
    if req.obj_height_m < 0:
        raise HTTPException(400, "obj_height_m ≥ 0")
    if not (-60 <= req.lat <= 60):
        raise HTTPException(400, "Copernicus ограничен ±60°")

    t0 = time.time()
    step = req.grid_step_m
    R = req.max_distance_km * 1000

    # 1) локальная UTM проекция
    utm_zone = int((req.lon + 180) // 6) + 1
    crs_utm = f"+proj=utm +zone={utm_zone} +datum=WGS84 +units=m +no_defs"
    to_utm = Transformer.from_crs(4326, crs_utm, always_xy=True)
    to_geo = Transformer.from_crs(crs_utm, 4326, always_xy=True)
    ox, oy = to_utm.transform(req.lon, req.lat)

    n = int(2 * R / step) + 1
    xs = np.linspace(ox - R, ox + R, n)
    ys = np.linspace(oy - R, oy + R, n)

    # 2) читаем DEM кусок, ресемплируем на нужный шаг
    lat_min, lon_min = to_geo.transform(xs[0], ys[-1])
    lat_max, lon_max = to_geo.transform(xs[-1], ys[0])
    fp = _path(int(math.floor(req.lat)), int(math.floor(req.lon)))
    with rasterio.open(fp) as ds:
        window = rasterio.windows.from_bounds(lon_min, lat_min, lon_max, lat_max,
                                              ds.transform)
        arr = ds.read(
            1,
            window=window,
            out_shape=(n, n),
            resampling=Resampling.bilinear
        ).astype(np.float32)
        arr[arr == ds.nodata] = 0

    obs_h = elev_vec(np.array([req.lon]), np.array([req.lat]))[0] + req.height_m
    r_eff = EARTH_RADIUS_M * req.k_factor

    visible = np.zeros_like(arr, dtype=np.bool_)

    # 3) LOS по строкам
    mid = n // 2
    for r in range(n):
        vis_r = _line_of_sight(arr[r, mid + 1:], obs_h, req.obj_height_m, r_eff, step)
        visible[r, mid + 1:][vis_r] = True
        vis_l = _line_of_sight(arr[r, mid - 1::-1], obs_h, req.obj_height_m, r_eff, step)
        visible[r, :mid][vis_l[::-1]] = True

    # 4) LOS по столбцам
    for c in range(n):
        vis_up = _line_of_sight(arr[mid - 1::-1, c], obs_h, req.obj_height_m, r_eff, step)
        visible[:mid, c][vis_up[::-1]] = True
        vis_dn = _line_of_sight(arr[mid + 1:, c], obs_h, req.obj_height_m, r_eff, step)
        visible[mid + 1:, c][vis_dn] = True

    vis_ratio = round(visible.mean() * 100, 2)
    log.info("mask ready in %.2f s, visible %.2f %%", time.time() - t0, vis_ratio)

    # 5) бинарный растр → полигоны
    trans = rasterio.transform.from_origin(xs[0] - step / 2, ys[-1] - step / 2,
                                           step, step)
    polys = []
    for geom, val in shapes(visible.astype(np.uint8), transform=trans):
        if val == 1:
            polys.append(shape(geom))
    union = unary_union(polys)
    geojson = mapping(union)

    return ViewshedResp(geojson=geojson, visible_ratio=vis_ratio)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("radio_viewshed_server:app", host="0.0.0.0", port=8011,
                log_level=LOG_LEVEL.lower())
