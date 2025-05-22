#!/usr/bin/env python3
"""
coverage_txfrac.py – учитывает передатчик, только если
  area(intersection) / area(tx) ≥ --min-tx-frac
"""
from __future__ import annotations
import argparse, json, hashlib, sys
from pathlib import Path
from typing import Any, List

import geopandas as gpd, pandas as pd, requests
from shapely.geometry import shape
from shapely.ops import transform
from pyproj import Transformer
import folium, branca.colormap as cm

CACHE = Path(".coverage_cache"); CACHE.mkdir(exist_ok=True)
WGS84 = "EPSG:4326"; WEBM = "EPSG:3857"
COLORS = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
          "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]

# ---------- helpers -------------------------------------------------
def _h(o:Any)->str: return hashlib.sha1(json.dumps(o,sort_keys=True).encode()).hexdigest()
def _swap(g):       return transform(lambda x,y,*a:(y,x,*a),g)
def _req(url:str,tx:dict):
    r=requests.post(f"{url}/radio_viewshed",json=tx,timeout=6000); r.raise_for_status()
    return shape(r.json()["geojson"])
def _req_cached(url:str,tx:dict):
    f=CACHE/f"{_h(tx)}.geojson"
    if f.exists(): return shape(json.load(f.open()))
    g=_req(url,tx); json.dump(g.__geo_interface__,f.open("w")); return g

def _is_longlat(g) -> bool:
    minx,miny,maxx,maxy = g.bounds
    return all(-180<=v<=180 for v in (minx,maxx)) and all(-90<=v<=90 for v in (miny,maxy))

# ---------- overlay -------------------------------------------------
def union_txfrac(gdfs:List[gpd.GeoDataFrame], min_frac:float)->gpd.GeoDataFrame:
    all_polys = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=WEBM)
    # атомарная «плитка»
    pieces = gpd.overlay(all_polys, all_polys, how="union", keep_geom_type=False)

    area_tx = all_polys.set_index("tx_id").geometry.area   # площадь каждого Tx
    pieces["n_tx"] = 0

    for tx_id, geom in zip(all_polys["tx_id"], all_polys.geometry):
        overlap = pieces.geometry.intersection(geom)
        mask = (overlap.area / area_tx[tx_id]) >= min_frac
        pieces.loc[mask, "n_tx"] += 1
    return pieces

# ---------- CLI / main ---------------------------------------------
def main():
    ap = argparse.ArgumentParser("Overlay with per-TX area threshold")
    ap.add_argument("tx_json"); ap.add_argument("--server", default="http://10.11.0.50:8011")
    ap.add_argument("--out", default="coverage_txfrac.html")
    ap.add_argument("--min-tx-frac", type=float, default=0.05, metavar="FRACTION")
    ap.add_argument("--swap-axes", action="store_true")
    args = ap.parse_args()

    txs = json.load(open(args.tx_json, encoding="utf-8"))
    gdfs: List[gpd.GeoDataFrame] = []
    to3857 = Transformer.from_crs(WGS84, WEBM, always_xy=True).transform

    for i, tx in enumerate(txs, 1):
        g = _req_cached(args.server, tx)
        if args.swap_axes: g = _swap(g)

        # если сервер дал градусы → переводим в метры
        if _is_longlat(g):
            g = transform(to3857, g)

        gdfs.append(gpd.GeoDataFrame({"tx_id":[i]}, geometry=[g], crs=WEBM))

    to4326 = Transformer.from_crs(WEBM, WGS84, always_xy=True).transform
    unioned = union_txfrac(gdfs, args.min_tx_frac)
    unioned["geometry"] = unioned.geometry.apply(lambda g: transform(to4326, g))
    unioned.set_crs(WGS84, allow_override=True, inplace=True)

    max_tx  = int(unioned["n_tx"].max()) or 1

    center  = list(unioned.unary_union.centroid.coords)[0][::-1]
    m = folium.Map(location=center, zoom_start=13)

    cmap = cm.linear.YlOrRd_09.scale(1, max_tx)
    folium.GeoJson(
        unioned.to_json(),
        style_function=lambda f: {
            "fillColor": cmap(f["properties"]["n_tx"]),
            "color":     "black",
            "weight":    0.4,
            "fillOpacity": 0.6},
        tooltip=folium.GeoJsonTooltip(fields=["n_tx"], aliases=["Передатчиков:"])
    ).add_to(m)
    cmap.caption = "n_tx"; cmap.add_to(m)

    for g,col in zip([g.to_crs(WGS84) for g in gdfs], COLORS*10):
        tid = int(g.iloc[0]["tx_id"])
        folium.GeoJson(g.to_json(), name=f"Tx {tid}",
            style_function=lambda _=None,c=col:{ "color":c, "weight":1, "fillOpacity":0.15 }
        ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(args.out)
    print("✔ Map saved:", args.out)

if __name__ == "__main__":
    main()
