#!/usr/bin/env python3
"""
coverage_overlay.py  — интерактивная карта перекрытия
=====================================================

* Запрашивает радиовидимость (viewshed) у **radio_horizon_server** для
  передатчиков и строит HTML‑карту.
* Каждый передатчик — отдельный слой (можно скрывать через LayerControl).
* Корректно работает, если сервер отдаёт полигон в:
    * WGS‑84 (EPSG:4326, градусы);
    * UTM‑зоне (метры) — определяется автоматически;
    * Web‑Mercator (EPSG:3857, метры).
* Если сервер перепутал lat↔lon, можно добавить `--swap-axes`.

Требует Python 3.9+, GeoPandas, Shapely ≥ 2.0, Folium, Branca, Requests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, List

import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
from shapely.ops import transform
import folium
import branca.colormap as cm

# ────────────────────────── constants ──────────────────────────
CACHE_DIR = Path(".coverage_cache")
WGS84_CRS = "EPSG:4326"
_TX_COLORS = [  # D3 Category10 (10 штук, затем повтор)
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

# ────────────────────────── helpers ───────────────────────────

def _tx_digest(tx: dict[str, Any]) -> str:
    """Хэш‑ключ по параметрам передатчика для кэша."""
    return hashlib.sha1(json.dumps(tx, sort_keys=True).encode()).hexdigest()


def _swap_latlon(geom):
    """Поменять X↔Y у всех координат."""
    return transform(lambda x, y, z=None: (y, x) if z is None else (y, x, z), geom)


def _guess_crs(geom) -> str:
    """Грубое определение системы координат по bbox."""
    minx, miny, maxx, maxy = geom.bounds
    # градусов
    if all(-180 <= v <= 180 for v in (minx, maxx)) and all(-90 <= v <= 90 for v in (miny, maxy)):
        return "EPSG:4326"
    # UTM‑полоса < 2000 км по X и метры по Y
    if abs(maxx) < 2_000_000 and abs(minx) < 2_000_000:
        return "UTM"
    # иначе считаем Web‑Mercator
    return "EPSG:3857"

# ───────────────────────── HTTP + cache ───────────────────────

def request_viewshed(server_url: str, tx: dict, timeout: int = 6033):
    """Запрос к серверу."""
    r = requests.post(f"{server_url}/radio_viewshed", json=tx, timeout=timeout)
    r.raise_for_status()
    return shape(r.json()["geojson"])  # shapely geometry


def request_viewshed_cached(server_url: str, tx: dict, timeout: int = 6033):
    CACHE_DIR.mkdir(exist_ok=True)
    fp = CACHE_DIR / f"{_tx_digest(tx)}.geojson"
    if fp.exists():
        with fp.open() as f:
            return shape(json.load(f))
    geom = request_viewshed(server_url, tx, timeout)
    with fp.open("w") as f:
        json.dump(geom.__geo_interface__, f)
    return geom

# ─────────────────────── geometry builders ────────────────────

def _utm_epsg(lon: float, lat: float) -> str:
    zone = int((lon + 180) // 6) + 1
    north = lat >= 0
    return f"EPSG:{32600 + zone if north else 32700 + zone}"


def build_coverages(
    transmitters: list[dict],
    server_url: str,
    input_crs_opt: str,
    swap_latlon: bool,
    debug: bool = False,
) -> List[gpd.GeoDataFrame]:
    """Скачать viewshed каждого передатчика и вернуть список GDF."""
    gdfs: List[gpd.GeoDataFrame] = []
    detected_crs: str | None = None

    for idx, tx in enumerate(transmitters, 1):
        geom = request_viewshed_cached(server_url, tx)

        # ручное принудительное переставление осей
        if swap_latlon:
            geom = _swap_latlon(geom)

        # выбор CRS: explicit → auto
        crs = input_crs_opt
        if input_crs_opt == "auto":
            if detected_crs is None:
                detected_crs = _guess_crs(geom)
            crs = detected_crs
        if crs == "UTM":
            crs = _utm_epsg(tx["lon"], tx["lat"])

        if debug:
            print(f"Tx {idx}: CRS={crs}, bounds={geom.bounds}", file=sys.stderr)

        gdfs.append(gpd.GeoDataFrame({"tx_id": [idx]}, geometry=[geom], crs=crs))
    return gdfs

# ─────────────────── overlay + counts ─────────────────────────

def union_with_counts(gdfs: List[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    if not gdfs:
        return gpd.GeoDataFrame(geometry=[], crs=WGS84_CRS)

    all_polys = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
    pieces = gpd.GeoDataFrame(geometry=[all_polys.unary_union], crs=all_polys.crs).explode(index_parts=False, ignore_index=True)
    joined = gpd.sjoin(pieces, all_polys[["geometry", "tx_id"]], predicate="intersects", how="left")
    pieces["n_tx"] = joined.groupby(joined.index)["tx_id"].nunique().astype(int)
    return pieces

# ──────────────────────── map rendering ───────────────────────

def _style_union(cm_):
    return lambda f: {"fillColor": cm_(f["properties"]["n_tx"]), "color": "black", "weight": 0.4, "fillOpacity": 0.6}


def save_html_map(gdfs: List[gpd.GeoDataFrame], unioned: gpd.GeoDataFrame, output: Path, zoom: int = 11):
    gdfs_w = [g.to_crs(WGS84_CRS) if g.crs != WGS84_CRS else g for g in gdfs]
    unioned_w = unioned.to_crs(WGS84_CRS) if unioned.crs != WGS84_CRS else unioned

    center = list(unioned_w.unary_union.centroid.coords)[0][::-1]  # lat, lon
    m = folium.Map(location=center, zoom_start=zoom)

    # общая тепловая заливка
    max_tx = max(unioned_w["n_tx"].max(), 1)
    cm_tx = cm.linear.YlOrRd_09.scale(1, int(max_tx))
    folium.GeoJson(unioned_w.to_json(), style_function=_style_union(cm_tx),
                   tooltip=folium.GeoJsonTooltip(fields=["n_tx"], aliases=["Передатчиков:"]))\
        .add_to(folium.FeatureGroup(name="Перекрытие (n_tx)").add_to(m))
    cm_tx.caption = "Кол-во передатчиков"
    cm_tx.add_to(m)

    # слои передатчиков
    for gdf, color in zip(gdfs_w, _TX_COLORS * 10):
        tx_id = int(gdf.iloc[0]["tx_id"])
        folium.GeoJson(gdf.to_json(), name=f"Tx {tx_id}",
                       style_function=lambda _=None, c=color: {"color": c, "weight": 1, "fillOpacity": 0.2},
                       tooltip=folium.GeoJsonTooltip(fields=["tx_id"], aliases=["Передатчик:"]))\
            .add_to(m)

    folium.LayerControl().add_to(m)
    m.save(output)
    print(f"Карта сохранена в {output}")

# ───────────────────────────── CLI ────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Интерактивная тепловая карта перекрытия радиопокрытий")
    p.add_argument("transmitters", help="JSON‑файл со списком передатчиков")
    p.add_argument("--server", default="http://10.11.0.50:8011", help="URL radio_horizon_server")
    p.add_argument("--out", default="coverage.html", help="Выходной HTML‑файл")
    p.add_argument("--zoom", type=int, default=11, help="Начальный масштаб карты")
    p.add_argument("--input-crs", default="auto", help="Явная проекция данных сервера или auto")
    p.add_argument("--swap-axes", action="store_true", help="Принудительно поменять lat↔lon")
    p.add_argument("--debug", action="store_true", help="Печатать CRS/bounds каждого слоя")
    return p.parse_args()

# ─────────────────────────── main ─────────────────────────────

def main():
    args = parse_args()
    with open(args.transmitters, "r", encoding="utf-8") as fp:
        transmitters = json.load(fp)

    gdfs = build_coverages(transmitters, args.server, args.input_crs, args.swap_axes, args.debug)
    unioned = union_with_counts(gdfs)
    save_html_map(gdfs, unioned, Path(args.out), zoom=args.zoom)

if __name__ == "__main__":
    main()
