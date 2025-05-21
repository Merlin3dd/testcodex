#!/usr/bin/env python3
'''
coverage_overlay.py

Запрашивает радиовидимость (viewshed) для набора передатчиков через
radio_horizon_server и строит карту зон перекрытия с подсветкой количества
видимых передатчиков.

Требует:
    - Python 3.9+
    - geopandas
    - shapely >= 2.0
    - requests
    - folium
    - branca

Пример запуска:
    python coverage_overlay.py transmitters.json --server http://localhost:8011 --out coverage.html
'''
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
import folium
import branca.colormap as cm


def request_viewshed(server_url: str, tx: dict, timeout: int = 6033):
    resp = requests.post(f'{server_url}/radio_viewshed', json=tx, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return shape(data['geojson'])


def build_coverages(transmitters: list[dict], server_url: str) -> list[gpd.GeoDataFrame]:
    gdfs: list[gpd.GeoDataFrame] = []
    for idx, tx in enumerate(transmitters, start=1):
        geom = request_viewshed(server_url, tx)
        gdf = gpd.GeoDataFrame(
            {'tx_id': [idx]},
            geometry=[geom],
            crs='EPSG:4326'
        ).explode(index_parts=False, ignore_index=True)
        gdfs.append(gdf)
    return gdfs


def union_with_counts(gdfs: list[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    all_polys = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs='EPSG:4326')
    unioned = gpd.overlay(all_polys, all_polys, how='union', keep_geom_type=False)

    # overlay создаёт _left_id и _right_id; считаем, сколько передатчиков покрывает фрагмент
    def _count(row):
        ids = {row['_left_id'], row.get('_right_id')}
        return sum(pd.notna(v) for v in ids)

    unioned['n_tx'] = unioned.apply(_count, axis=1)
    return unioned


def save_html_map(unioned: gpd.GeoDataFrame, output_file: Path, center: list[float] | None = None, zoom: int = 11):
    if unioned.empty:
        print('Нет данных для построения карты', file=sys.stderr)
        return
    if center is None:
        centroid = unioned.unary_union.centroid
        center = [centroid.y, centroid.x]

    m = folium.Map(location=center, zoom_start=zoom)

    max_tx = int(unioned['n_tx'].max())
    colormap = cm.linear.YlOrRd_09.scale(1, max_tx)

    folium.GeoJson(
        unioned.to_json(),
        style_function=lambda f: {
            'fillColor': colormap(f['properties']['n_tx']),
            'color': 'black',
            'weight': 0.5,
            'fillOpacity': 0.6,
        },
        tooltip=folium.GeoJsonTooltip(fields=['n_tx'], aliases=['Передатчиков:'])
    ).add_to(m)

    colormap.caption = 'Количество передатчиков'
    colormap.add_to(m)
    m.save(output_file)
    print(f'Карта сохранена в {output_file}')


def parse_args():
    p = argparse.ArgumentParser(description='Генерация тепловой карты перекрытия радиопокрытий')
    p.add_argument('transmitters', help='JSON‑файл со списком передатчиков')
    p.add_argument('--server', default='http://10.11.0.50:8011',
                   help='URL radio_horizon_server (по умолчанию: %(default)s)')
    p.add_argument('--out', default='coverage.html', help='Имя выходного HTML‑файла')
    p.add_argument('--zoom', type=int, default=11, help='Начальный масштаб')
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.transmitters, 'r', encoding='utf-8') as fp:
        transmitters = json.load(fp)
    gdfs = build_coverages(transmitters, args.server)
    unioned = union_with_counts(gdfs)
    save_html_map(unioned, Path(args.out), zoom=args.zoom)


if __name__ == '__main__':
    main()
