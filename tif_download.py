#!/usr/bin/env python3
"""
get_copdem_fast.py – быстрая закачка Copernicus DEM 10 m
• Параллельно качает тысячи тайлов, насыщая канал 1 Gb/s
• Контроль целостности: MD5 (локально) ↔ ETag (на S3)
• Показывает общий прогресс и текущую скорость в Mb/s

Использование
    python get_copdem_fast.py LAT_MAX LON_MIN LAT_MIN LON_MAX OUT_DIR [WORKERS]

Пример
    python get_copdem_fast.py 77.939609 -8.734777 30.179682 102.626224 dem_raw
"""

import math
import sys
import pathlib
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
from botocore import UNSIGNED
from botocore.client import Config
from boto3.s3.transfer import TransferConfig
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────── аргументы
if len(sys.argv) < 6:
    sys.exit("usage: get_copdem_fast.py LAT_MAX LON_MIN LAT_MIN LON_MAX OUT_DIR [WORKERS]")

lat_max, lon_min, lat_min, lon_max = map(float, sys.argv[1:5])
out_dir   = pathlib.Path(sys.argv[5]).expanduser()
workers   = int(sys.argv[6]) if len(sys.argv) > 6 else 8         # 8 потоков ≈ 1 Gb/s
out_dir.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────── helpers
def tag(v: float, pos: str, width: int) -> str:
    """55 → N55_00   |  -9 → W009_00"""
    return f"{pos}{abs(int(v)):0{width}d}_00"

def md5sum(path: pathlib.Path, buf=1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(buf):
            h.update(chunk)
    return h.hexdigest()

# ─────────────────────────────────────────────────────────────────── диапазон тайлов
lat_rg = range(math.floor(lat_min), math.ceil(lat_max))
lon_rg = range(math.floor(lon_min), math.ceil(lon_max))

tiles = [
    f"Copernicus_DSM_COG_10_{tag(lat,'N' if lat >= 0 else 'S',2)}_"
    f"{tag(lon,'E' if lon >= 0 else 'W',3)}_DEM"
    for lat in lat_rg
    for lon in lon_rg
]

# ─────────────────────────────────────────────────────────────────── S3‑клиент
s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

xfer_cfg = TransferConfig(
    multipart_threshold = 8 * 1024 * 1024,     # ≥ 8 МБ → multi‑part
    multipart_chunksize = 8 * 1024 * 1024,     # 8 МБ части
    max_concurrency     = 10,                  # потоков на файл
    use_threads         = True,
)

# ─────────────────────────────────────────────────────────────────── MD5 ↔ ETag
def etag_matches(tile: str, local: pathlib.Path) -> bool:
    """True, если локальный MD5 совпал с ETag.  Для одиночных загрузок ETag == MD5."""
    try:
        head = s3.head_object(
            Bucket="copernicus-dem-30m",
            Key=f"{tile}/{tile}.tif",
            ChecksumMode="ENABLED",
        )
        etag = head["ETag"].strip('"')
        if "-" in etag:          # multipart (не бывает для 40 МБ, но на всякий случай)
            return True
        return md5sum(local) == etag
    except Exception:
        return False

# ─────────────────────────────────────────────────────────────────── прогресс‑бар
BYTES_PER_TILE = 40.3 * 1024 * 1024          # грубо 40 МиБ
bar   = tqdm(total=int(len(tiles) * BYTES_PER_TILE),
             unit="B", unit_scale=True, unit_divisor=1024,
             desc="Total", dynamic_ncols=True)
lock  = threading.Lock()                     # tqdm не потокобезопасен

# ─────────────────────────────────────────────────────────────────── загрузка тайла
def download(tile: str) -> str:
    dst = out_dir / f"{tile}.tif"

    # файл уже есть и контроль пройден
    if dst.exists() and etag_matches(tile, dst):
        with lock:
            bar.update(dst.stat().st_size)
        return f"✓ {tile} (ok)"

    transferred = 0

    # callback считает байты; bar.update() должен быть под блокировкой
    def cb(bytes_amount: int):
        nonlocal transferred
        transferred += bytes_amount
        with lock:
            bar.update(bytes_amount)

    try:
        s3.download_file(
            Bucket   = "copernicus-dem-30m",
            Key      = f"{tile}/{tile}.tif",
            Filename = str(dst),
            Config   = xfer_cfg,
            Callback = cb,
        )
        if etag_matches(tile, dst):
            return f"✓ {tile}"
        dst.unlink(missing_ok=True)
        return f"✗ {tile}: checksum mismatch (deleted)"
    except Exception as e:
        dst.unlink(missing_ok=True)
        # компенсируем «лишние» байты, если ошибка раньше конца файла
        with lock:
            bar.update(-transferred)
        return f"✗ {tile}: {e}"

# ─────────────────────────────────────────────────────────────────── старт
print(f"Start download: {len(tiles)} tiles, {workers} workers …", flush=True)
with ThreadPoolExecutor(max_workers=workers) as pool:
    for fut in as_completed(pool.submit(download, t) for t in tiles):
        print(fut.result())

bar.close()
print("Done.")