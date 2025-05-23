#!/usr/bin/env python3
"""
coverage_txfrac_async.py – ускоренная версия «coverage_txfrac»
-----------------------------------------------------------
Основные изменения
• ***Параллельные HTTP‑запросы***: httpx.AsyncClient + asyncio (по умолчанию 16 одновременных).
• ***Кэш***: прежний on‑disk .coverage_cache остаётся, так что повторные запросы не идут к серверу.
• ***CLI***: добавлены аргументы `--concurrency` и `--max-connections`.
• ***Оверлей***: прежний `union_txfrac` (со spatial‑index) сохранён; его можно заменить
  на более быстрый vectorised‑вариант, см. комментарии в коде.

Требует: Python ≥3.8, httpx, geopandas, shapely ≥1.8, folium, branca.

Запуск:
    python coverage_txfrac_async.py tx.json --server http://10.11.0.50:8011

Совет: на сервере запустите `uvicorn radio_viewshed_server:app --workers 8` —
тогда параллельные запросы действительно распределятся по CPU‑ядрам.
"""
import argparse, asyncio, hashlib, json, sys, time
from pathlib import Path
from typing import Any, List

import orjson, numpy as np  # ← добавили сюда
import geopandas as gpd, pandas as pd, httpx
import rasterio
from shapely.geometry import shape
from shapely.ops import transform
from pyproj import Transformer
import folium, branca.colormap as cm
from tqdm import tqdm
import rasterio
import rasterio.features as rio_features
# ───────────────────────── constants ──────────────────────────
ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".coverage_cache"
CACHE.mkdir(exist_ok=True)
PARCELS_DIR = ROOT / "parcels"
WGS84 = "EPSG:4326"
WEBM = "EPSG:3857"
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


# ───────────────────────── helpers ────────────────────────────

def find_parcels(dir_path: Path = PARCELS_DIR) -> Path:
    """Return newest GeoJSON/GPKG file from ``dir_path``."""
    candidates = [p for p in dir_path.glob("*")
                  if p.suffix.lower() in {".geojson", ".gpkg"}]
    if not candidates:
        raise FileNotFoundError(f"No parcels file found in {dir_path}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _h(o: Any) -> str:
    """
    SHA-1 от компактного ORJSON-представления объекта.
    orjson.dumps быстрее стандартного json.dumps в 3-4 раза.
    """
    return hashlib.sha1(orjson.dumps(o, option=orjson.OPT_SORT_KEYS)).hexdigest()


def _swap(g):
    """Поменять местами оси (lon↔lat) — полезно для ошибочных GeoJSON."""
    return transform(lambda x, y, *a: (y, x, *a), g)


def _is_longlat(g) -> bool:
    minx, miny, maxx, maxy = g.bounds
    return all(-180 <= v <= 180 for v in (minx, maxx)) and all(-90 <= v <= 90 for v in (miny, maxy))


# ───────────────────────── async fetch ─────────────────────────
HEADERS = {"accept-encoding": "br, gzip"}  # можно вынести в constants


async def _fetch_viewshed(
        session: httpx.AsyncClient,
        url: str,
        tx: dict,
        tx_id: int,
        swap_axes: bool,
        to3857,
) -> gpd.GeoDataFrame:
    """
    Читает viewshed из кеша либо запрашивает у сервера и кладёт в кеш.
    Возвращает GeoDataFrame в проекции WebMercator (EPSG:3857).
    """
    f = CACHE / f"{_h(tx)}.geojson"

    try:  # быстрее, чем Path.exists()
        g = shape(json.load(f.open()))
    except FileNotFoundError:  # нет в кеше → fetch
        r = await session.post(f"{url}/radio_viewshed", json=tx)
        r.raise_for_status()
        g = shape(r.json()["geojson"])
        json.dump(g.__geo_interface__, f.open("w"))

    if swap_axes:
        g = _swap(g)
    if _is_longlat(g):  # координаты в WGS-84?
        g = transform(to3857, g)

    return gpd.GeoDataFrame({"tx_id": [tx_id]}, geometry=[g], crs=WEBM)


async def gather_viewsheds(
        server: str,
        txs: List[dict],
        swap_axes: bool,
        concurrency: int,
        max_conn: int,
) -> List[gpd.GeoDataFrame]:
    """
    Параллельно получает все viewshed-полигоны с учётом реального ограничения
    одновременных запросов (Semaphore). Клиент работает поверх HTTP/2.
    """
    limits = httpx.Limits(max_connections=max_conn)
    timeout = httpx.Timeout(600.0)
    to3857 = Transformer.from_crs(WGS84, WEBM, always_xy=True).transform
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(
            http2=True, limits=limits, timeout=timeout, headers=HEADERS
    ) as session:
        async def bound_fetch(tx, idx):
            async with sem:
                return await _fetch_viewshed(session, server, tx, idx, swap_axes, to3857)

        tasks = [asyncio.create_task(bound_fetch(tx, i + 1))
                 for i, tx in enumerate(txs)]
        results = []
        for coro in tqdm(asyncio.as_completed(tasks),
                         total=len(tasks),
                         desc="viewsheds",
                         unit="tx"):
            results.append(await coro)
        return results


# ───────────────────────── geometry utils ──────────────────────

def union_txfrac_vec(
        gdfs: list[gpd.GeoDataFrame],
        min_frac: float = 0.05
) -> gpd.GeoDataFrame:
    """
    Overlay 'union' всех viewshed-полигонов и подсчёт,
    сколько TX перекрывают ≥ min_frac площади САМОГО кусочка.
    """
    all_polys = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=WEBM)
    pieces = gpd.overlay(all_polys, all_polys, how="union", keep_geom_type=False)

    sidx = pieces.sindex
    if hasattr(sidx, "query_bulk"):                           # Shapely-2 / PyGEOS
        idx_tx, idx_piece = sidx.query_bulk(                  # 0-й — TX, 1-й — piece
            all_polys.geometry, predicate="intersects"
        )
    else:                                                     # GeoPandas ≥ 0.15
        res = sidx.query(all_polys.geometry, predicate="intersects")
        if isinstance(res, np.ndarray):
            idx_tx, idx_piece = res
        else:
            idx_tx, idx_piece = res.iloc[:, 0].to_numpy(), res.iloc[:, 1].to_numpy()

    s_piece = pieces.geometry.take(idx_piece).reset_index(drop=True)
    s_tx    = all_polys.geometry.take(idx_tx).reset_index(drop=True)

    inter_area = s_piece.intersection(s_tx, align=False).area.to_numpy()
    piece_area = pieces.area.to_numpy()[idx_piece]            # доля от куска
    area_ratio = inter_area / piece_area

    n_tx = np.bincount(idx_piece[area_ratio >= min_frac], minlength=len(pieces))
    pieces["n_tx"] = n_tx
    return pieces


def tx_count_per_parcel(
        parcels: gpd.GeoDataFrame,
        tx_gdfs: list[gpd.GeoDataFrame],
        min_cover_frac: float = 0.05
) -> gpd.GeoDataFrame:
    """
    Для каждого полигона parcel считает, сколько viewshed-полигонов
    перекрывают ≥ min_cover_frac площади ЭТОГО parcel.

    Возвращает тот же GeoDataFrame (crs сохранится) + колонка `n_tx`.
    """
    parcels = parcels.to_crs(WEBM).copy()
    parcels["p_area"] = parcels.area

    tx_all = gpd.GeoDataFrame(pd.concat(tx_gdfs, ignore_index=True), crs=WEBM)

    sidx = tx_all.sindex
    ix_parcel, ix_tx = sidx.query_bulk(parcels.geometry, predicate="intersects")

    g_parcel = parcels.geometry.take(ix_parcel).reset_index(drop=True)
    g_tx = tx_all.geometry.take(ix_tx).reset_index(drop=True)

    inter_area = g_parcel.intersection(g_tx, align=False).area.to_numpy()
    parcel_area = parcels["p_area"].values[ix_parcel]
    good = inter_area / parcel_area >= min_cover_frac

    n_tx = np.bincount(ix_parcel[good], minlength=len(parcels))
    parcels["n_tx"] = n_tx

    return parcels.drop(columns="p_area")
def cover_raster_sum(
        gdfs: list[gpd.GeoDataFrame],
        pixel_m: float = 10.0
) -> tuple[np.ndarray, rasterio.transform.Affine]:
    all_bounds = np.vstack([g.total_bounds for g in gdfs])
    minx, miny = all_bounds[:, 0].min(), all_bounds[:, 1].min()
    maxx, maxy = all_bounds[:, 2].max(), all_bounds[:, 3].max()

    nx = int(np.ceil((maxx - minx) / pixel_m))
    ny = int(np.ceil((maxy - miny) / pixel_m))
    transform = rasterio.transform.from_origin(minx, maxy, pixel_m, pixel_m)

    cover = np.zeros((ny, nx), dtype=np.uint16)

    for g in gdfs:
        shapes_iter = ((geom, 1) for geom in g.geometry)
        rio_features.rasterize(                 # ← используем импортированный alias
            shapes_iter,
            out=cover,
            transform=transform,
            merge_alg=rasterio.enums.MergeAlg.add,
            fill=0,
        )
    return cover, transform

def raster_to_geojson(cover: np.ndarray, transform) -> gpd.GeoDataFrame:
    """0-значения отбрасываем, остальные превращаем в полигоны с n_tx."""
    shapes_iter = rasterio.features.shapes(
        cover, mask=cover > 0, transform=transform
    )
    records = [(shape(geom), int(val)) for geom, val in shapes_iter]
    gdf = gpd.GeoDataFrame(
        {"n_tx": [v for _, v in records]},
        geometry=[g for g, _ in records],
        crs=WEBM
    )
    return gdf
# ───────────────────────── main CLI ────────────────────────────
def main():
    ap = argparse.ArgumentParser("Overlay with per-TX area threshold (async)")
    ap.add_argument("tx_json", help="Файл с массивом JSON-объектов запросов к /radio_viewshed")
    ap.add_argument("--server", default="http://10.11.0.50:8011",
                    help="URL FastAPI-сервера")
    ap.add_argument("--out", default="coverage_txfrac.html",
                    help="HTML-файл карты")
    ap.add_argument("--min-tx-frac", type=float, default=0.05, metavar="FRACTION",
                    help="Мин. доля площади передатчика, пересекающая кусок (старый режим)")
    ap.add_argument("--swap-axes", action="store_true",
                    help="Поменять lon/lat в GeoJSON ответа")
    ap.add_argument("--concurrency", type=int, default=16, metavar="N",
                    help="Сколько viewshed-запросов слать одновременно")
    ap.add_argument("--max-connections", type=int, default=32, metavar="M",
                    help="Предел одновременных TCP-соединений httpx-клиента")
    ap.add_argument("--parcels", help="GeoJSON / GeoPackage с полигонами участков")
    ap.add_argument("--min-parcel-frac", type=float, default=0.05,
                    help="Мин. доля площади участка, покрытая TX (новый режим)")
    ap.add_argument("--mode", choices=["union", "parcel", "raster"],
                    default="union",
                    help="union = overlay кусков (бывш. по умолчанию); "
                         "parcel = внешний слой участков; "
                         "raster = бинарная маска + суммирование")
    ap.add_argument("--pixel-m", type=float, default=10,
                    help="Размер пикселя (м) для raster-режима")

    args = ap.parse_args()
    t0 = time.perf_counter()

    # ── читаем список TX-запросов ───────────────────────────────
    txs = json.load(open(args.tx_json, encoding="utf-8"))
    if not isinstance(txs, list):
        print("✘ Ожидается JSON-массив объектов", file=sys.stderr)
        sys.exit(1)

    # ── параллельно получаем GeoDataFrame-ы --───────────────────
    gdfs = asyncio.run(
        gather_viewsheds(args.server, txs, args.swap_axes,
                         args.concurrency, args.max_connections)
    )

    # ── «участковый» или «старый» режим визуализации ────────────
    t_cov = time.perf_counter()
    if args.mode == "parcel":
        if not args.parcels:
            args.parcels = find_parcels()
            print(f"✔ Using parcels: {args.parcels}")
        parcels = gpd.read_file(args.parcels)
        gdf_vis = tx_count_per_parcel(
            parcels, gdfs, min_cover_frac=args.min_parcel_frac
        )

    elif args.mode == "raster":
        cover, aff = cover_raster_sum(gdfs, pixel_m=args.pixel_m)
        gdf_vis = raster_to_geojson(cover, aff)

    else:  # union
        gdf_vis = union_txfrac_vec(gdfs, args.min_tx_frac)

    dt_cov = time.perf_counter() - t_cov
    print(f"\u2714 Coverage ({args.mode}) computed in {dt_cov:.1f}s")

    # ── переводим итоговый слой в WGS-84 ───────────────────────────
    to4326 = Transformer.from_crs(WEBM, WGS84, always_xy=True).transform
    gdf_vis["geometry"] = gdf_vis.geometry.apply(lambda g: transform(to4326, g))
    gdf_vis.set_crs(WGS84, allow_override=True, inplace=True)

    # ── Folium визуализация (одна для всех режимов) ───────────────
    max_tx = int(gdf_vis["n_tx"].max()) or 1
    center = list(gdf_vis.unary_union.centroid.coords)[0][::-1]
    m = folium.Map(location=center, zoom_start=13)

    cmap = cm.linear.YlOrRd_09.scale(1, max_tx)
    folium.GeoJson(
        gdf_vis.to_json(),
        name="coverage",
        style_function=lambda f: {
            "fillColor": cmap(f["properties"]["n_tx"]),
            "color": "black",
            "weight": 0.4,
            "fillOpacity": 0.6,
        },
        tooltip=folium.GeoJsonTooltip(fields=["n_tx"],
                                      aliases=["Передатчиков:"])
    ).add_to(m)

    # ── общие элементы (легенда, слои TX, LayerControl, save) ──
    cmap.caption = "n_tx"
    cmap.add_to(m)

    # слои по каждому TX (для дебага)
    for g, col in zip([g.to_crs(WGS84) for g in gdfs], COLORS * 10):
        tid = int(g.iloc[0]["tx_id"])
        folium.GeoJson(
            g.to_json(), name=f"Tx {tid}",
            style_function=lambda _=None, c=col: {
                "color": c, "weight": 1, "fillOpacity": 0.15
            }
        ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(args.out)
    dt = time.perf_counter() - t0
    print(f"✔ Map saved: {args.out} in {dt:.1f}s")


if __name__ == "__main__":
    main()
