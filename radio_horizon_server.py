"""
radio_viewshed_server.py  ·  Copernicus DSM 10 m
POST /radio_viewshed → GeoJSON MultiPolygon с видимыми пятнами
"""

from __future__ import annotations
import hashlib, math, os, pathlib, time, logging, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple, List
import rasterio.merge, rasterio.warp
import numpy as np, rasterio, requests
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.features import shapes
from shapely.geometry import shape, mapping
from shapely.ops import unary_union, transform
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pyproj import Geod, Transformer
from tqdm import tqdm
from numba import njit
from osgeo import gdal
import json

DEM_CACHE_DIR = pathlib.Path(".dem_cache")
DEM_CACHE_DIR.mkdir(exist_ok=True)

gdal.UseExceptions()
# ──────────────────────── logging ────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL,
                    format="%(asctime)s [%(levelname)7s] %(name)s: %(message)s")
log = logging.getLogger("viewshed")

# ──────────────────────── constants ──────────────────────
COP_DIR = pathlib.Path(os.getenv(
    "COP_DIR",
    "/home/user/backend/maps/work/dem_raw/"
)).expanduser()

COP_DIR.mkdir(parents=True, exist_ok=True)

EARTH_RADIUS_M = 6_371_000.0
geod = Geod(ellps="WGS84")

log.debug(COP_DIR)

GEBCO_ZIP = COP_DIR / "gebco_2024_geotiff.zip"  # архив 7 ГБ
GEBCO_SRC = COP_DIR / "gebco_2024_geotiff"  # 8 файлов 90×90°
GEBCO_TILES = COP_DIR / "gebco_tiles_10d"  # целевые 10×10° COG
_GEBCO_READY = False  # уже нарезано?
_GEBCO_LOCK = threading.Lock()  # гарантируем один проход


# ──────────────────────── DEM loader ─────────────────────
def _key(lat: int, lon: int, res: int = 10) -> str:
    """
    Вернуть относительный ключ Copernicus DEM для широты/долготы.
    res — разрешение (10 или 30 м).
    """
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    stem = f"Copernicus_DSM_COG_{res}_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"
    return f"{stem}/{stem}.tif"


def _path(lat: int, lon: int) -> pathlib.Path:
    return COP_DIR / pathlib.Path(_key(lat, lon)).name


def _ensure_gebco_tiles():
    """
    Гарантирует, что 10°×10° COG-тайлы GEBCO готовы.
    Функция потокобезопасна: нарезка выполняется однократно.
    """
    global _GEBCO_READY

    # быстрый выход, если уже готово
    if _GEBCO_READY:
        return

    # единственный поток проходит внутрь, остальные ждут
    with _GEBCO_LOCK:
        if _GEBCO_READY:  # проверка ещё раз — мог сделать другой поток
            return

        # ── 1) архив ───────────────────────────────
        if not GEBCO_ZIP.exists():
            url = ("https://www.bodc.ac.uk/data/open_download/"
                   "gebco/gebco_2024/geotiff/gebco_2024_geotiff.zip")
            _fetch_to_cache(url, GEBCO_ZIP.name)

        # ── 2) распаковка ──────────────────────────
        if not (GEBCO_SRC.exists() and any(GEBCO_SRC.glob("*.tif"))):
            import zipfile
            GEBCO_SRC.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(GEBCO_ZIP) as zf:
                zf.extractall(GEBCO_SRC)

        # ── 3) тайлы 10×10° ────────────────────────
        if len(list(GEBCO_TILES.glob("*.tif"))) != 648:
            GEBCO_TILES.mkdir(parents=True, exist_ok=True)

            log.info("⧗ Разовая нарезка GEBCO → 10×10° COG (~1–2 мин)")
            px = 1 / 240.0  # 15″

            for src_path in sorted(GEBCO_SRC.glob("*.tif")):
                ds = gdal.Open(str(src_path))
                gt = ds.GetGeoTransform()
                xmin, ymax = gt[0], gt[3]
                xmax = xmin + gt[1] * ds.RasterXSize
                ymin = ymax + gt[5] * ds.RasterYSize

                def round10(val):  # -180 … +180
                    return int(round(val / 10.0)) * 10

                lat_min = round10(ymin)
                lat_max = round10(ymax)
                lon_min = round10(xmin)
                lon_max = round10(xmax)

                for lat0 in range(lat_min, lat_max, 10):
                    for lon0 in range(lon_min, lon_max, 10):
                        lat1, lon1 = lat0 + 10, lon0 + 10

                        ns = "n" if lat0 >= 0 else "s"
                        ew = "w" if lon0 < 0 else "e"
                        tif = f"gebco_2024_{ns}{abs(lat0):02d}_{ew}{abs(lon0):03d}_10d.tif"
                        dst = GEBCO_TILES / tif
                        if dst.exists():
                            continue

                        opts = gdal.TranslateOptions(
                            format="COG",
                            creationOptions=["COMPRESS=DEFLATE",
                                             "PREDICTOR=2",
                                             "BLOCKSIZE=512"],
                            projWin=[lon0, lat1, lon1, lat0],  # W N E S
                            xRes=px, yRes=px,
                            resampleAlg="average"
                        )
                        gdal.Translate(str(dst), ds, options=opts)
                ds = None
            log.info("✓ GEBCO COG-тайлы готовы")

        # раз и навсегда помечаем
        _GEBCO_READY = True


def _md5(fname: pathlib.Path, buf: int = 1 << 20) -> str:
    h = hashlib.md5()
    with fname.open("rb") as f:
        chunk = f.read(buf)
        while chunk:
            h.update(chunk)
            chunk = f.read(buf)
    return h.hexdigest()


def _fetch_to_cache(url: str, name: str) -> pathlib.Path:
    """
    Качает url в COP_DIR/name.
    • Если файл уже в кеше и MD5/ETag совпадает ― скачивание пропускается.
    • Всегда пишет в лог ПОЛНЫЙ путь к файлу.
    """
    dst = COP_DIR / name
    log.info("[START] %s → %s", url, dst)  # ← полный путь

    # ── HEAD ───────────────────────────────────────────────
    etag, total = "", None
    try:
        head = requests.head(url, timeout=20, allow_redirects=True)
        if head.status_code == 200:
            etag = head.headers.get("ETag", "").strip('"')
            total = int(head.headers.get("Content-Length", 0) or 0)
            log.debug("[HEAD ] 200 OK  size=%s  etag=%s", total, etag[:8])
        elif head.status_code == 403:
            log.debug("[HEAD ] 403 AccessDenied – пропускаем")
        else:
            head.raise_for_status()
    except requests.RequestException as e:
        log.debug("[HEAD ] %s – %s", type(e).__name__, e)

    # ── cache-hit ──────────────────────────────────────────
    if dst.exists() and (etag == "" or "-" in etag or _md5(dst) == etag):
        log.info("[CACHE] hit  %s", dst)  # ← полный путь
        return dst

    # ── GET ────────────────────────────────────────────────
    log.info("[GET  ] → %s", dst)  # ← полный путь
    r = requests.get(url, stream=True, timeout=120)
    r.raise_for_status()
    if total is None:
        total = int(r.headers.get("Content-Length", 0) or 0)

    bar = tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024,
               desc=f"↓ {dst.name}", leave=False)
    next_pct = 5
    t0 = time.time()
    tmp = dst.with_suffix(".part")

    with tmp.open("wb") as f:
        for chunk in r.iter_content(1 << 15):
            f.write(chunk)
            bar.update(len(chunk))
            if total and bar.n >= total * next_pct / 100:
                log.debug("[GET  ] %s %2d%%", dst.name, next_pct)
                next_pct += 5
    bar.close()

    if etag and "-" not in etag and _md5(tmp) != etag:
        tmp.unlink()
        raise IOError("MD5 mismatch after download")

    tmp.rename(dst)
    dt = time.time() - t0
    mb = total / 1_048_576 if total else 0
    sp = mb / dt if dt else 0
    log.info("[DONE ] %s  %.1f MiB in %.1fs (%.1f MiB/s)",
             dst, mb, dt, sp)  # ← полный путь
    return dst


def _open_tile(lat: int, lon: int):
    """
    Открывает и возвращает rasterio.DatasetReader для тайла 1×1°.
    После успешного открытия пишет в лог полный путь (ds.name).
    """
    try:
        ds = rasterio.open(_download(lat, lon))
    except IOError:
        oce = _download_ocean(lat, lon)
        if oce:
            ds = rasterio.open(oce)
        else:
            tf = rasterio.transform.from_bounds(lon, lat, lon + 1, lat + 1, 1200, 1200)
            ds = _zero_dataset(1200, 1200, tf).open()

    log.debug("Tile %-2d,%-3d ← %s", lat, lon, ds.name)  # ← полный путь
    return ds


# ──────────────────────── ocean DEM ──────────────────────
def _download_ocean(lat: int, lon: int) -> str | None:
    """
    Возвращает путь к 10×10° COG-тайлу для данной клетки (lat, lon).
    Если файлов ещё нет, запускает их создание.
    """
    _ensure_gebco_tiles()  # гарантирует наличие COG-тайлов

    lat0 = (lat // 10) * 10
    lon0 = (lon // 10) * 10
    ns = "n" if lat0 >= 0 else "s"
    ew = "w" if lon0 < 0 else "e"
    tif = f"gebco_2024_{ns}{abs(lat0):02d}_{ew}{abs(lon0):03d}_10d.tif"

    path = GEBCO_TILES / tif
    return str(path) if path.exists() else None


# ──────────────────────── Copernicus DEM ─────────────────
def _download(lat: int, lon: int) -> pathlib.Path:
    """
    Скачивает Copernicus DEM (10 м или 30 м). 10 м – если есть доступ;
    30 м – открытый слой. Бросает IOError, если отсутствуют оба.
    """
    for res, bucket in [(10, "https://copernicus-dem-10m.s3.amazonaws.com"),
                        (30, "https://copernicus-dem-30m.s3.amazonaws.com")]:
        k = _key(lat, lon, res)
        url = f"{bucket}/{k}"
        log.info("[SRC  ] Copernicus %2d m → %s", res, k)

        try:
            head = requests.head(url, timeout=20, allow_redirects=True)
        except requests.RequestException as e:
            log.debug("[HEAD ] %s – %s", url, e)
            continue

        if head.status_code == 200:
            return _fetch_to_cache(url, pathlib.Path(k).name)
        if head.status_code == 403 and res == 30:
            return _fetch_to_cache(url, pathlib.Path(k).name)
        if head.status_code not in (401, 403, 404):
            head.raise_for_status()

    raise IOError(f"DEM tile for {lat},{lon} not found in 10 m or 30 m archives")


class Tile:
    __slots__ = ("arr", "inv", "_mem")

    def __init__(self, lat: int, lon: int):
        """
        1) Copernicus 10 м → 30 м
        2) GEBCO 2024 (из world-zip)
        3) единичная «плашка» 0 м
        """
        try:
            fp = _download(lat, lon)
            ds = rasterio.open(fp)
            self._mem = None
        except IOError:
            oce = _download_ocean(lat, lon)
            if oce:
                ds = rasterio.open(oce)
                self._mem = None
                log.debug("Fallback to GEBCO %s,%s", lat, lon)
            else:
                tf = rasterio.transform.from_bounds(lon, lat, lon + 1, lat + 1, 1, 1)
                mem = MemoryFile()
                with mem.open(driver="GTiff", height=1, width=1, count=1,
                              dtype="float32", crs="EPSG:4326",
                              transform=tf, nodata=0) as tmp:
                    tmp.write(np.zeros((1, 1, 1), dtype=np.float32))
                ds = mem.open()
                self._mem = mem

        with ds:
            self.arr = ds.read(1).astype(np.float32)
            self.arr[self.arr == ds.nodata] = 0
            self.inv = ~ds.transform
        log.debug("Tile %s,%s loaded  shape=%s", lat, lon, self.arr.shape)

    # ← вернулся потерянный метод
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
    k_factor: float = Field(4 / 3)


def _zero_dataset(rows, cols, transform):
    """Создаёт in-memory GeoTIFF из нулей, который понимает rasterio.open()."""
    profile = dict(
        driver="GTiff",
        height=rows, width=cols,
        count=1, dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=0,
        compress="deflate"
    )
    mem = MemoryFile()
    with mem.open(**profile) as ds:
        ds.write(np.zeros((1, rows, cols), dtype=np.float32))
    return mem


class ViewshedResp(BaseModel):
    geojson: dict
    visible_ratio: float


app = FastAPI(title="Copernicus viewshed", version="1.0")


@app.get("/health")
async def health(): return {"ok": True}


# ──────────────────────── main algorithm ─────────────────
def dem_cache_key(lat_min, lon_min, lat_max, lon_max, out_shape, resampling):
    # Ключ: только значимые параметры!
    o = {
        "lat_min": round(float(lat_min), 5),
        "lon_min": round(float(lon_min), 5),
        "lat_max": round(float(lat_max), 5),
        "lon_max": round(float(lon_max), 5),
        "out_shape": list(out_shape),
        "resampling": str(resampling),
    }
    return hashlib.sha1(json.dumps(o, sort_keys=True).encode()).hexdigest()


def load_dem_from_cache(lat_min, lon_min, lat_max, lon_max, out_shape, resampling):
    key = dem_cache_key(lat_min, lon_min, lat_max, lon_max, out_shape, resampling)
    path = DEM_CACHE_DIR / f"{key}.npz"
    if path.exists():
        npz = np.load(path)
        arr = npz["arr"]
        trans = npz["trans"]
        return arr, trans
    return None, None


def save_dem_to_cache(lat_min, lon_min, lat_max, lon_max, out_shape, resampling, arr, trans):
    key = dem_cache_key(lat_min, lon_min, lat_max, lon_max, out_shape, resampling)
    path = DEM_CACHE_DIR / f"{key}.npz"
    np.savez_compressed(path, arr=arr, trans=trans)


def mosaic_dem(lat_min, lon_min, lat_max, lon_max, out_shape, resampling):
    """
    Собирает мозаичный DEM для заданного окна.
    • Тайлы 1×1° скачиваются/читаются ПАРАЛЛЕЛЬНО (ThreadPoolExecutor).
    • После сборки все datasets закрываются.
    """
    arr, trans = load_dem_from_cache(lat_min, lon_min, lat_max, lon_max, out_shape, resampling)
    if arr is not None and trans is not None:
        return arr, trans

    lat_idxs = range(int(math.floor(lat_min)), int(math.ceil(lat_max)))
    lon_idxs = range(int(math.floor(lon_min)), int(math.ceil(lon_max)))
    todo = [(la, lo) for la in lat_idxs for lo in lon_idxs]

    # ① параллельно загружаем все тайлы
    srcs: List[rasterio.DatasetReader] = []
    workers = min(8, len(todo)) or 1  # 8 потоков максимум
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_open_tile, la, lo): (la, lo) for la, lo in todo}
        for fut in as_completed(futures):
            try:
                srcs.append(fut.result())
            except Exception as exc:
                la, lo = futures[fut]
                log.error("Tile %s,%s failed: %s", la, lo, exc)
                raise

    # ② мержим в единую корзину
    mosaic, trans0 = rasterio.merge.merge(srcs, resampling=Resampling.nearest)

    # обязательно закрываем все открытые datasets
    for ds in srcs:
        ds.close()

    # ③ ресэмплинг в нужный grid
    dst_rows, dst_cols = out_shape
    dst = np.empty(out_shape, dtype=np.float32)
    dst_trans = rasterio.transform.from_bounds(
        lon_min, lat_min, lon_max, lat_max, dst_cols, dst_rows
    )

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
    save_dem_to_cache(lat_min, lon_min, lat_max, lon_max, out_shape, resampling, dst, dst_trans)

    return dst, dst_trans


from numba import njit, prange


@njit(fastmath=True, parallel=True, cache=True)
def _viewshed_360(arr, step, obs_h, obj_h, r_eff):
    """
    Быстрый круговой viewshed: arr — квадратная DEM (n×n), наблюдатель в центре.
    Возвращает bool-маску видимости.
    """
    n = arr.shape[0]
    mid = n // 2
    R = (n - 1) * step / 2.0
    dθ = math.atan(step / R)  # минимальный угол, чтобы зацепить клетку
    n_ang = int(2 * math.pi / dθ) + 1

    visible = np.zeros((n, n), np.bool_)

    for a in prange(n_ang):
        ang = a * dθ
        dx = math.cos(ang)
        dy = -math.sin(ang)  # y растёт вниз

        max_t = -1e20
        for i in range(1, n):
            xf = mid + dx * i
            yf = mid + dy * i
            xi = int(xf)
            yi = int(yf)
            if xi < 0 or yi < 0 or xi >= n or yi >= n:
                break  # вышли за пределы квадрата

            s = i * step
            tan = (arr[yi, xi] + obj_h - (s * s) / (2 * r_eff) - obs_h) / s
            if tan > max_t:  # клетка видна
                max_t = tan
                visible[yi, xi] = True
    return visible


@app.post("/radio_viewshed", response_model=ViewshedResp)
async def radio_viewshed(req: ViewshedReq):
    t_start = time.time()

    # Проверки
    if req.obj_height_m < 0:
        raise HTTPException(400, "obj_height_m ≥ 0")
    if not (-60 <= req.lat <= 60):
        raise HTTPException(400, "Copernicus ограничен ±60°")

    t0 = time.time()
    step = req.grid_step_m  # шаг DEM (м)
    R = req.max_distance_km * 1_000  # радиус расчёта (м)

    # Локальная UTM-проекция
    utm_zone = int((req.lon + 180) // 6) + 1
    crs_utm = f"+proj=utm +zone={utm_zone} +datum=WGS84 +units=m +no_defs"
    to_utm = Transformer.from_crs(4326, crs_utm, always_xy=True)
    to_geo = Transformer.from_crs(crs_utm, 4326, always_xy=True)
    ox, oy = to_utm.transform(req.lon, req.lat)  # центр в метрах

    # Квадратная матрица DEM вокруг точки
    n = int(2 * R / step) + 1  # кол-во пикселей
    xs = np.linspace(ox - R, ox + R, n)  # UTM-X
    ys = np.linspace(oy - R, oy + R, n)  # UTM-Y

    lon_w, lat_n = to_geo.transform(xs[0], ys[-1])  # NW-угол
    lon_e, lat_s = to_geo.transform(xs[-1], ys[0])  # SE-угол
    lon_min, lon_max = sorted((lon_w, lon_e))
    lat_min, lat_max = sorted((lat_s, lat_n))

    t1 = time.time()
    arr, _ = mosaic_dem(lat_min, lon_min, lat_max, lon_max,
                        out_shape=(n, n),
                        resampling=Resampling.bilinear)
    t2 = time.time()

    obs_h = elev_vec(np.array([req.lon]), np.array([req.lat]))[0] + req.height_m
    r_eff = EARTH_RADIUS_M * req.k_factor  # эффективный радиус

    t3 = time.time()
    visible = _viewshed_360(arr, step, obs_h, req.obj_height_m, r_eff)
    t4 = time.time()

    trans = rasterio.transform.from_origin(xs[0] - step / 2,
                                           ys[-1] - step / 2,
                                           step, step)
    polys_utm = [shape(geom) for geom, val in
                 shapes(visible.astype(np.uint8), transform=trans) if val == 1]
    to4326 = Transformer.from_crs(crs_utm, "EPSG:4326",
                                  always_xy=True).transform
    polys_ll = [transform(to4326, p) for p in polys_utm]
    geojson = mapping(unary_union(polys_ll))
    vis_ratio = round(visible.mean() * 100, 2)
    t5 = time.time()

    # Логгируем каждый этап
    log.info("⏱️ Этапы расчёта viewshed:")
    log.info("  1. Pre-proc (UTM/coords):    %.2fs", t1 - t0)
    log.info("  2. DEM mosaic+resample:      %.2fs", t2 - t1)
    log.info("  3. LOS подготовка:           %.2fs", t3 - t2)
    log.info("  4. Viewshed Numba:           %.2fs", t4 - t3)
    log.info("  5. Mask→GeoJSON (shapes):    %.2fs", t5 - t4)
    log.info("  ∑ Всего времени запроса:     %.2fs", t5 - t_start)

    return ViewshedResp(geojson=geojson, visible_ratio=vis_ratio)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("radio_viewshed_server:app", host="0.0.0.0", port=8011,
                log_level=LOG_LEVEL.lower())
