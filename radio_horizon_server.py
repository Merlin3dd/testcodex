"""
radio_viewshed_server.py  ·  Copernicus DSM 10 m
POST /radio_viewshed → GeoJSON MultiPolygon с видимыми пятнами
"""

from __future__ import annotations
import hashlib, math, os, pathlib, time, logging, threading, concurrent.futures
from typing import Dict, Tuple, List
import rasterio, rasterio.merge, rasterio.warp
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

COP_DIR = pathlib.Path(os.getenv("COP_DIR", "/home/user/work/dem_raw/")).expanduser()
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
@njit(fastmath=True, parallel=True, cache=True)
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
def mosaic_dem(lat_min, lon_min, lat_max, lon_max, out_shape, resampling):
    """
    Склеить все тайлы, перекрывающие рамку, и ресемплировать
    в out_shape (n_rows, n_cols).  Возвращает (array, transform).
    """
    lats = range(int(math.floor(lat_min)), int(math.ceil(lat_max)))
    lons = range(int(math.floor(lon_min)), int(math.ceil(lon_max)))
    srcs = [
        rasterio.open(_download(la, lo))  # гарантированно существует
        for la in lats for lo in lons
    ]

    # ① склейка без ресэмпла, в исходной геодезической проекции
    mosaic, trans0 = rasterio.merge.merge(srcs)
    for s in srcs:
        s.close()

    # ② задаём целевой grid
    dst_rows, dst_cols = out_shape
    dst = np.empty(out_shape, dtype=np.float32)
    dst_trans = rasterio.transform.from_bounds(
        lon_min, lat_min, lon_max, lat_max, dst_cols, dst_rows
    )

    # ③ reproject → сразу Bilinear, nodata→0
    rasterio.warp.reproject(
        source=mosaic[0],
        destination=dst,
        src_transform=trans0,
        src_crs="EPSG:4326",
        dst_transform=dst_trans,
        dst_crs="EPSG:4326",
        resampling=resampling,
    )
    dst[np.isnan(dst)] = 0
    return dst, dst_trans
from numba import njit, prange

@njit(fastmath=True, parallel=True, cache=True)
def _viewshed_360(arr, step, obs_h, obj_h, r_eff):
    """
    Быстрый круговой viewshed: arr — квадратная DEM (n×n), наблюдатель в центре.
    Возвращает bool-маску видимости.
    """
    n   = arr.shape[0]
    mid = n // 2
    R   = (n - 1) * step / 2.0
    dθ  = math.atan(step / R)            # минимальный угол, чтобы зацепить клетку
    n_ang = int(2 * math.pi / dθ) + 1

    visible = np.zeros((n, n), np.bool_)

    for a in prange(n_ang):
        ang = a * dθ
        dx  =  math.cos(ang)
        dy  = -math.sin(ang)             # y растёт вниз

        max_t = -1e20
        for i in range(1, n):
            xf = mid + dx * i
            yf = mid + dy * i
            xi = int(xf)
            yi = int(yf)
            if xi < 0 or yi < 0 or xi >= n or yi >= n:
                break                    # вышли за пределы квадрата

            s   = i * step
            tan = (arr[yi, xi] + obj_h - (s * s) / (2 * r_eff) - obs_h) / s
            if tan > max_t:              # клетка видна
                max_t      = tan
                visible[yi, xi] = True
    return visible
@app.post("/radio_viewshed", response_model=ViewshedResp)
async def radio_viewshed(req: ViewshedReq):
    # ────── проверки входных данных ──────────────────────────────
    if req.obj_height_m < 0:
        raise HTTPException(400, "obj_height_m ≥ 0")
    if not (-60 <= req.lat <= 60):
        raise HTTPException(400, "Copernicus ограничен ±60°")

    t0   = time.time()
    step = req.grid_step_m                      # шаг DEM (м)
    R    = req.max_distance_km * 1_000          # радиус расчёта (м)

    # ────── локальная UTM-проекция ───────────────────────────────
    utm_zone = int((req.lon + 180) // 6) + 1
    crs_utm  = f"+proj=utm +zone={utm_zone} +datum=WGS84 +units=m +no_defs"
    to_utm   = Transformer.from_crs(4326, crs_utm, always_xy=True)
    to_geo   = Transformer.from_crs(crs_utm, 4326, always_xy=True)
    ox, oy   = to_utm.transform(req.lon, req.lat)        # центр в метрах

    # ────── квадратная матрица DEM вокруг точки ─────────────────
    n  = int(2 * R / step) + 1                          # кол-во пикселей
    xs = np.linspace(ox - R, ox + R, n)                 # UTM-X
    ys = np.linspace(oy - R, oy + R, n)                 # UTM-Y

    lon_w, lat_n = to_geo.transform(xs[0],   ys[-1])    # NW-угол
    lon_e, lat_s = to_geo.transform(xs[-1],  ys[0])     # SE-угол
    lon_min, lon_max = sorted((lon_w, lon_e))
    lat_min, lat_max = sorted((lat_s, lat_n))

    arr, _ = mosaic_dem(lat_min, lon_min, lat_max, lon_max,
                        out_shape=(n, n),
                        resampling=Resampling.bilinear)

    # ────── подготовка высот/радиусов ────────────────────────────
    obs_h = elev_vec(np.array([req.lon]), np.array([req.lat]))[0] + req.height_m
    r_eff = EARTH_RADIUS_M * req.k_factor            # эффективный радиус

    visible = np.zeros_like(arr, dtype=np.bool_)     # итоговая маска
    mid     = n // 2                                 # индекс наблюдателя

    # ────── РАДИАЛЬНЫЙ ПРОХОД ПО ВСЕМ АЗИМУТАМ ───────────────────
    # угловой шаг: чтобы луч «зацепил» каждую ячейку
    visible = _viewshed_360(arr, step, obs_h, req.obj_height_m, r_eff)
    vis_ratio = round(visible.mean() * 100, 2)


    # ────── бинарная маска → GeoJSON-мультиполигон ───────────────
    trans = rasterio.transform.from_origin(xs[0] - step / 2,
                                           ys[-1] - step / 2,
                                           step, step)
    polys = [shape(geom) for geom, val in
             shapes(visible.astype(np.uint8), transform=trans) if val == 1]
    geojson = mapping(unary_union(polys))

    return ViewshedResp(geojson=geojson, visible_ratio=vis_ratio)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("radio_viewshed_server:app", host="0.0.0.0", port=8011,
                log_level=LOG_LEVEL.lower())
