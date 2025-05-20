from __future__ import annotations
import gzip, io, math, os, threading, zipfile, hashlib, pathlib, time, logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import boto3, numpy as np, rasterio, requests
from botocore import UNSIGNED
from botocore.client import Config
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pyproj import Geod
from shapely.geometry import LineString, mapping

# ──────────────────────────────  logging  ──────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("horizon")

# ────────────────────────────  константы  ──────────────────────────────
EARTH_RADIUS_M = 6_371_000.0
geod = Geod(ellps="WGS84")
NODATA = -32768

# ───────────────────  Copernicus DEM — локальный и S3  ──────────────────
COP_DIR = pathlib.Path(os.getenv("COP_DIR", "./copernicus"))
COP_DIR.mkdir(parents=True, exist_ok=True)
log.info("COP_DIR  = %s", COP_DIR.resolve())

s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

def cop_path(lat:int, lon:int)->pathlib.Path:
    ns = "N" if lat>=0 else "S"; ew = "E" if lon>=0 else "W"
    return COP_DIR/f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM.tif"

def md5(path: pathlib.Path, buf=1<<20)->str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()

def download_cop(lat:int, lon:int)->pathlib.Path:
    dst = cop_path(lat, lon)
    key = f"{dst.stem}/{dst.stem}.tif"

    if dst.exists():
        etag = s3.head_object(Bucket="copernicus-dem-30m", Key=key)["ETag"].strip('"')
        if "-" in etag or md5(dst) == etag:
            log.debug("Copernicus tile %s already cached", dst.name)
            return dst
        log.warning("Copernicus MD5 mismatch -> re-download %s", dst.name)
        dst.unlink(missing_ok=True)

    tmp = dst.with_suffix(".part")
    log.info("Downloading Copernicus %s …", dst.name)
    t0 = time.time()
    s3.download_file("copernicus-dem-30m", key, str(tmp))
    t1 = time.time()
    speed = tmp.stat().st_size / (t1 - t0) / 1024 / 1024
    log.info("Copernicus %s downloaded (%.1f MiB, %.1f MB/s)",
             dst.name, tmp.stat().st_size/1048576, speed)

    etag = s3.head_object(Bucket="copernicus-dem-30m", Key=key)["ETag"].strip('"')
    if "-" not in etag and md5(tmp) != etag:
        tmp.unlink(missing_ok=True)
        raise IOError("MD5 mismatch on Copernicus tile")
    tmp.rename(dst)
    return dst

class CopTile:
    def __init__(self, lat:int, lon:int):
        self.lat, self.lon = lat, lon
        self.path = download_cop(lat, lon)
        self.ds   = rasterio.open(self.path)
        log.debug("Opened Copernicus raster %s", self.path.name)
    def elevation(self, x:float, y:float)->float:
        val = next(self.ds.sample([(x,y)]))[0]
        return float(val) if val != self.ds.nodata else 0.0

class CopManager:
    def __init__(self):
        self.tiles: Dict[Tuple[int,int], CopTile|None] = {}
        self.lock = threading.Lock()
    def _tile(self, lon:float, lat:float):
        key = (int(math.floor(lat)), int(math.floor(lon)))
        with self.lock:
            if key not in self.tiles:
                try:
                    self.tiles[key] = CopTile(*key)
                except Exception as e:
                    log.warning("Copernicus tile %s unavailable: %s", key, e)
                    self.tiles[key] = None
            return self.tiles[key]
    def elevation(self, lon:float, lat:float)->float|None:
        tile = self._tile(lon, lat)
        return tile.elevation(lon, lat) if tile else None
COP = CopManager()

# ────────────────────────────  SRTM (.hgt)  ────────────────────────────
SRTM_DIR = pathlib.Path(os.getenv("SRTM_DIR", "./srtm_cache"))
SRTM_DIR.mkdir(parents=True, exist_ok=True)
log.info("SRTM_DIR = %s", SRTM_DIR.resolve())

MAPZEN = "https://s3.amazonaws.com/elevation-tiles-prod/skadi/{t}/{t}.hgt.gz"
ESA    = "https://step.esa.int/auxdata/dem/SRTMGL1/{t}.SRTMGL1.hgt.zip"

def tname(lat:int, lon:int)->str:
    return f"{'N' if lat>=0 else 'S'}{abs(lat):02d}{'E' if lon>=0 else 'W'}{abs(lon):03d}"

def urls(tile:str): return [(MAPZEN.format(t=tile),"gz"), (ESA.format(t=tile),"zip")]

class SrtmTile:
    def __init__(self, lat:int, lon:int):
        self.lat, self.lon = lat, lon
        self.tile = tname(lat, lon)
        self.hgt  = SRTM_DIR/f"{self.tile}.hgt"
        self.arr  = None
        self.samples = None
        self.lock = threading.Lock()

    def _download(self):
        for url, kind in urls(self.tile):
            try:
                r = requests.get(url, timeout=30)
                if r.status_code == 404:
                    continue
                r.raise_for_status()
                log.info("Downloading SRTM %s from %s", self.tile, url.split('/')[2])
                if kind == "gz":
                    with gzip.GzipFile(fileobj=io.BytesIO(r.content)) as g, open(self.hgt,"wb") as o:
                        o.write(g.read())
                else:
                    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                        with z.open(z.namelist()[0]) as src, open(self.hgt,"wb") as dst:
                            dst.write(src.read())
                return
            except requests.RequestException as e:
                log.debug("Failed %s: %s", url, e)
        raise FileNotFoundError(self.tile)

    def _load(self):
        if self.arr is not None:
            return
        with self.lock:
            if self.arr is not None:
                return
            if not self.hgt.exists():
                self._download()
            data = np.fromfile(self.hgt, dtype=">i2").astype(np.float32)
            self.samples = int(round(math.sqrt(data.size)))
            self.arr = data.reshape((self.samples, self.samples))
            self.arr[self.arr == NODATA] = np.nan
            log.debug("Loaded SRTM %s (%dx%d)", self.tile, self.samples, self.samples)

    def elev(self, lon:float, lat:float)->float:
        self._load()
        row = int((self.lat + 1 - lat)*(self.samples-1))
        col = int((lon - self.lon)*(self.samples-1))
        row, col = np.clip(row, 0, self.samples-1), np.clip(col, 0, self.samples-1)
        v = self.arr[row, col]
        return float(v) if not np.isnan(v) else 0.

class SrtmManager:
    def __init__(self):
        self.tiles: Dict[Tuple[int,int], SrtmTile] = {}
        self.lock = threading.Lock()
    def _tile(self, lon, lat):
        key = (int(math.floor(lat)), int(math.floor(lon)))
        with self.lock:
            if key not in self.tiles:
                self.tiles[key] = SrtmTile(*key)
            return self.tiles[key]
    def elevation(self, lon, lat): return self._tile(lon, lat).elev(lon, lat)
SRTM = SrtmManager()

# ──────────────────────────  единая функция высоты  ─────────────────────────
def elevation(lon:float, lat:float)->float:
    h = COP.elevation(lon, lat)
    source = "COP" if h is not None else "SRTM"
    val = h if h is not None else SRTM.elevation(lon, lat)
    log.debug("Elevation at (%.4f, %.4f) = %.1f via %s", lat, lon, val, source)
    return val

# ────────────────────────────  FastAPI слой  ───────────────────────────────
class Req(BaseModel):
    lat:float; lon:float; height_m:float
    max_distance_km:float=100; azimuth_step_deg:float=1
    sample_step_m:float=100; k_factor:float=Field(4/3)
class Resp(BaseModel):
    geojson:dict; points:List[Tuple[float,float]]

app = FastAPI(title="Radio Horizon API", version="3.2")

@app.get("/health")
async def health(): return {"status":"ok"}

def trace(az, req:Req, o_msl, r_eff, max_d):
    s, best, max_ang = req.sample_step_m, (req.lon, req.lat), -math.inf
    while s <= max_d:
        lon2, lat2, _ = geod.fwd(req.lon, req.lat, az, s)
        g = elevation(lon2, lat2)
        ang = math.atan2(g - (s**2)/(2*r_eff) - o_msl, s)
        if ang > max_ang:
            max_ang, best = ang, (lon2, lat2)
        s += req.sample_step_m
    return best

def horizon(req:Req):
    log.info("Start horizon calc lat=%.5f lon=%.5f h=%.1f m …",
             req.lat, req.lon, req.height_m)
    t0 = time.time()
    obs_msl = elevation(req.lon, req.lat) + req.height_m
    r_eff, max_d = EARTH_RADIUS_M*req.k_factor, req.max_distance_km*1000
    az = np.arange(0, 360, req.azimuth_step_deg)
    pts = [None]*len(az)
    with ThreadPoolExecutor() as pool:
        fut = {pool.submit(trace, a, req, obs_msl, r_eff, max_d): i
               for i, a in enumerate(az)}
        for f in as_completed(fut):
            pts[fut[f]] = f.result()
    dt = time.time() - t0
    log.info("Horizon calc done in %.2f s (%.1f azimuths)", dt, len(az))
    return pts

@app.post("/radio_horizon", response_model=Resp)
async def radio(req:Req):
    if not(-60 <= req.lat <= 60):
        raise HTTPException(400, "DEM покрывает ±60°")
    if not(0 < req.azimuth_step_deg <= 10):
        raise HTTPException(400, "azimuth_step_deg (0,10]")
    if req.sample_step_m <= 0:
        raise HTTPException(400, "sample_step_m > 0")
    pts = horizon(req); ls = LineString(pts + [pts[0]])
    return Resp(geojson={"type":"Feature","geometry":mapping(ls),
                         "properties":req.dict()}, points=pts)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("radio_horizon_server:app",
                host="0.0.0.0", port=8011,
                log_level=LOG_LEVEL.lower())