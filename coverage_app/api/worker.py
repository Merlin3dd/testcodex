from pathlib import Path
import asyncio, json, uuid, time
from shapely.ops import transform          # ← было забыто
from coverage_app.core.coverage_async import (
    gather_viewsheds, union_txfrac_vec,
    tx_count_per_parcel, cover_raster_sum,
    raster_to_geojson, find_parcels
)
import geopandas as gpd
from pyproj import Transformer
import folium, branca.colormap as cm
import rasterio
import rasterio.features as rio_features   # ← ДОБАВИТЬ
WEBM, WGS84 = "EPSG:3857", "EPSG:4326"
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]

async def run_coverage_async(
    tx_json_path: Path,
    server: str = "http://10.11.0.50:8011",
    out_dir: Path = Path(__file__).resolve().parents[2] / "static/maps",
    out_name: str | None = None,
    swap_axes: bool = False,
    concurrency: int = 16,
    max_connections: int = 32,
    mode: str = "union",
    min_tx_frac: float = .05,
    parcels_path: Path | None = None,
    min_parcel_frac: float = .05,
    pixel_m: float = 10.,
) -> str:
    """
    Выполняет полный расчёт покрытия и сохраняет интерактивную карту.
    Возвращает относительный URL вида ``/static/maps/<file>.html``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if not out_name:
        out_name = f"coverage_{uuid.uuid4().hex[:8]}.html"
    out_file = out_dir / out_name

    # --- читаем входной JSON --------------------------------------------------
    txs = json.load(tx_json_path.open("r", encoding="utf-8"))

    # --- получаем все viewshed-слои ------------------------------------------
    gdfs = await gather_viewsheds(
        server, txs, swap_axes, concurrency, max_connections
    )

    # --- формируем итоговый слой ---------------------------------------------
    t_cov = time.perf_counter()
    if mode == "parcel":
        if not parcels_path:
            parcels_path = find_parcels()
            print(f"✔ Using parcels: {parcels_path}")
        parcels = gpd.read_file(parcels_path)
        gdf_vis = tx_count_per_parcel(parcels, gdfs, min_parcel_frac)

    elif mode == "raster":
        cover, aff = cover_raster_sum(gdfs, pixel_m)
        gdf_vis = raster_to_geojson(cover, aff)

    else:                               # union
        gdf_vis = union_txfrac_vec(gdfs, min_tx_frac)

    dt_cov = time.perf_counter() - t_cov
    print(f"\u2714 Coverage ({mode}) computed in {dt_cov:.1f}s")

    # --- перевод в WGS-84 -----------------------------------------------------
    to4326 = Transformer.from_crs(WEBM, WGS84, always_xy=True).transform
    gdf_vis["geometry"] = gdf_vis.geometry.apply(lambda g: transform(to4326, g))
    gdf_vis.set_crs(WGS84, allow_override=True, inplace=True)

    # --- строим карту ---------------------------------------------------------
    max_tx = int(gdf_vis["n_tx"].max()) or 1
    center = list(gdf_vis.unary_union.centroid.coords)[0][::-1]
    m = folium.Map(location=center, zoom_start=13)
    cmap = cm.linear.YlOrRd_09.scale(1, max_tx)

    # итоговый слой покрытия  ⬇⬇⬇
    folium.GeoJson(
        gdf_vis.to_json(), name="coverage",
        style_function=lambda f: {
            "fillColor": cmap(f["properties"]["n_tx"]),
            "color": "black", "weight": .4, "fillOpacity": .6,
        },
        tooltip=folium.GeoJsonTooltip(fields=["n_tx"], aliases=["TX:"]),
    ).add_to(m)

    # debug-слои по каждому TX
    for g, col in zip([g.to_crs(WGS84) for g in gdfs], COLORS * 10):
        tid = int(g.iloc[0]["tx_id"])
        folium.GeoJson(
            g.to_json(), name=f"Tx {tid}",
            style_function=lambda _, c=col: {
                "color": c, "weight": 1, "fillOpacity": .15,
            },
        ).add_to(m)

    cmap.caption = "n_tx"
    cmap.add_to(m)
    folium.LayerControl().add_to(m)

    m.save(out_file)
    return f"/static/maps/{out_name}"
