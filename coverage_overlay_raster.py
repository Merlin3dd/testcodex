#!/usr/bin/env python3
"""
coverage_raster.py – тепловая карта по ячейкам
"""
from __future__ import annotations
import argparse, json, hashlib, io, base64, sys
from pathlib import Path
from typing import Any, List

import numpy as np, requests, rasterio, geopandas as gpd
from rasterio.features import rasterize
from shapely.geometry import shape
from shapely.ops import transform
from pyproj import Transformer
from PIL import Image
import folium, branca.colormap as cm

CACHE = Path(".coverage_cache");
CACHE.mkdir(exist_ok=True)
WGS84 = "EPSG:4326";
WEBM = "EPSG:3857"


def _hash(o: Any) -> str: return hashlib.sha1(json.dumps(o, sort_keys=True).encode()).hexdigest()


def _swap(g):       return transform(lambda x, y, *a: (y, x, *a), g)


def _req(url: str, tx: dict):
    r = requests.post(f"{url}/radio_viewshed", json=tx, timeout=6000);
    r.raise_for_status()
    return shape(r.json()["geojson"])


def _req_cached(url: str, tx: dict):
    f = CACHE / f"{_hash(tx)}.geojson"
    if f.exists(): return shape(json.load(f.open()))
    g = _req(url, tx);
    json.dump(g.__geo_interface__, f.open("w"));
    return g


def _is_longlat(g) -> bool:
    minx, miny, maxx, maxy = g.bounds
    return all(-180 <= v <= 180 for v in (minx, maxx)) and all(-90 <= v <= 90 for v in (miny, maxy))


def _hex2rgba(h: str) -> tuple[int, int, int, int]:
    h = h.lstrip("#");
    r = int(h[0:2], 16);
    g = int(h[2:4], 16);
    b = int(h[4:6], 16)
    return r, g, b, 255


def main():
    ap = argparse.ArgumentParser("Raster coverage map")
    ap.add_argument("tx_json");
    ap.add_argument("--server", default="http://10.11.0.50:8011")
    ap.add_argument("--cell-size", type=float, default=20, help="Шаг сетки, м")
    ap.add_argument("--out", default="coverage_raster.html")
    ap.add_argument("--swap-axes", action="store_true");
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    txs = json.load(open(args.tx_json, encoding="utf-8"))
    geoms = [_req_cached(args.server, tx) for tx in txs]
    if args.swap_axes: geoms = [_swap(g) for g in geoms]

    to3857 = Transformer.from_crs(WGS84, WEBM, always_xy=True).transform
    geoms_m = [transform(to3857, g) if _is_longlat(g) else g for g in geoms]
    gs = gpd.GeoSeries(geoms_m, crs=WEBM)

    minx, miny, maxx, maxy = gs.total_bounds
    res = args.cell_size
    width, height = int(np.ceil((maxx - minx) / res)), int(np.ceil((maxy - miny) / res))

    if args.debug: print(f"Grid {width}×{height}  step {res}m", file=sys.stderr)

    transform_r = rasterio.transform.from_origin(minx, maxy, res, res)
    accum = np.zeros((height, width), np.uint16)

    for g in geoms_m:
        mask = rasterize([(g, 1)], out_shape=accum.shape, transform=transform_r, all_touched=True)
        accum += mask.astype(np.uint16)

    max_tx = int(accum.max())
    if max_tx == 0:
        print("‼  Пустая сетка – проверьте координаты", file=sys.stderr);
        sys.exit(1)

    # --- PNG ---------------------------------------------------
    cmap = cm.linear.YlOrRd_09.scale(1, max_tx)
    img = np.zeros((*accum.shape, 4), np.uint8)
    for k in range(1, max_tx + 1):
        img[accum == k] = _hex2rgba(cmap(k))
    png = Image.fromarray(img)
    buf = io.BytesIO()
    png.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    # --- углы растра (WGS84) ----------------------------------
    to4326 = Transformer.from_crs(WEBM, WGS84, always_xy=True).transform
    lon1, lat1 = to4326(minx, miny)
    lon2, lat2 = to4326(maxx, maxy)

    m = folium.Map(location=[(lat1 + lat2) / 2, (lon1 + lon2) / 2], zoom_start=13)
    folium.raster_layers.ImageOverlay(
        image=f"data:image/png;base64,{img_b64}",
        bounds=[[lat1, lon1], [lat2, lon2]], opacity=0.6, name="n_tx raster"
    ).add_to(m)
    cmap.caption = "Кол-во передатчиков"
    cmap.add_to(m)
    folium.LayerControl().add_to(m)
    m.save(args.out)
    print("✔ Map saved:", args.out)


if __name__ == "__main__": main()
