#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
join_convert_npytoTIF.py
========================
Junta patches _pred.npy por região×ano em um único GeoTIFF.

Estrutura esperada de entrada:
  <predict_dir>/<region_id>/<year>/patch_r<row>_c<col>_pred.npy
  <predict_dir>/<region_id>/<year>/patch_r<row>_c<col>_pred.json

Saída (pasta plana):
  <output_dir>/pred_<region_id[:-5]>_<year>.tif

Uso:
  python join_convert_npytoTIF.py \\
      --predict-dir ~/db_images/predict_fotovoltaica \\
      --output-dir  ~/db_images/tif_fotovoltaica \\
      --years 2022 2023 \\
      --regions R_0001 R_0002
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

LOG_FILE = Path('join_convert_npyTIF.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
    ],
)
log = logging.getLogger(__name__)

PATCH_SIZE = 256


def _load_patches(npy_files: list[Path]) -> list[tuple]:
    patches = []
    for npy_path in npy_files:
        json_path = npy_path.with_suffix('.json')
        if not json_path.exists():
            log.warning(f'  JSON ausente: {json_path.name}')
            continue
        try:
            arr  = np.load(npy_path).astype(np.float32)
            meta = json.loads(json_path.read_text(encoding='utf-8'))
            patches.append((arr, meta))
        except Exception as exc:
            log.error(f'  Erro ao ler {npy_path.name}: {exc}')
    return patches


def _mosaic_rowcol(patches: list[tuple]) -> tuple[np.ndarray, Affine, str]:
    """Formato npy original: metadados com row, col, transform, crs."""
    def get_meta(row, col):
        return next((m for _, m in patches if m['row'] == row and m['col'] == col), None)

    m00 = get_meta(0, 0)
    stride = PATCH_SIZE
    if m00 is not None:
        m01 = get_meta(0, 1)
        if m01:
            stride = round((m01['transform'][2] - m00['transform'][2]) / m00['transform'][0])
        else:
            m10 = get_meta(1, 0)
            if m10:
                stride = round(abs(m10['transform'][5] - m00['transform'][5]) / abs(m00['transform'][4]))

    max_row = max(m['row'] for _, m in patches)
    max_col = max(m['col'] for _, m in patches)
    log.info(f'    stride={stride}px  grid {max_row+1}×{max_col+1}  patches={len(patches)}')

    h = max_row * stride + PATCH_SIZE
    w = max_col * stride + PATCH_SIZE
    mosaic  = np.zeros((h, w), dtype=np.float32)
    weights = np.zeros((h, w), dtype=np.float32)

    for arr, meta in patches:
        r, c = meta['row'], meta['col']
        y0, x0 = r * stride, c * stride
        mosaic [y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE] += arr
        weights[y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE] += 1.0

    weights = np.where(weights == 0, 1.0, weights)
    mosaic  = mosaic / weights

    t      = next(m for _, m in patches if m['row'] == 0 and m['col'] == 0)['transform']
    affine = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
    crs    = next(m for _, m in patches if m['row'] == 0 and m['col'] == 0).get('crs', 'EPSG:4326')
    return mosaic, affine, crs


def _cluster_coords(values: list[float], tol: float = 1e-4) -> list[float]:
    """
    Agrupa coordenadas que distam menos de `tol` graus (ruído de ponto flutuante
    entre patches da mesma linha/coluna). Retorna o centroide de cada cluster.
    tol=1e-4° ≈ 11 m — muito menor que o espaçamento real entre linhas (~0.011°).
    """
    sorted_v = sorted(values)
    clusters: list[list[float]] = [[sorted_v[0]]]
    for v in sorted_v[1:]:
        if v - clusters[-1][-1] < tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [sum(c) / len(c) for c in clusters]


def _mosaic_latlon(patches: list[tuple], pixel_size_m: float = 4.77) -> tuple[np.ndarray, Affine, str]:
    """
    Formato gee-tfrecord: metadados com latitude/longitude (centro do patch).
    pixel_size_m: resolução espacial em metros (Planet NICFI ≈ 4.77 m).
    Estima tamanho do pixel em graus a partir do espaçamento entre linhas/colunas
    reais de patches (depois de agrupar coordenadas com ruído de ponto flutuante).
    """
    import math

    lats = [m['latitude']  for _, m in patches]
    lons = [m['longitude'] for _, m in patches]

    mid_lat = sum(lats) / len(lats)

    # Agrupa lats/lons por proximidade para eliminar ruído de ponto flutuante
    row_lats = sorted(_cluster_coords(lats), reverse=True)   # N → S
    col_lons = sorted(_cluster_coords(lons))                  # W → E

    if len(row_lats) >= 2:
        spacing_lat = min(abs(a - b) for a, b in zip(row_lats, row_lats[1:]))
        px_lat = spacing_lat / PATCH_SIZE
    else:
        px_lat = pixel_size_m / 111_320.0

    if len(col_lons) >= 2:
        spacing_lon = min(abs(a - b) for a, b in zip(col_lons, col_lons[1:]))
        px_lon = spacing_lon / PATCH_SIZE
    else:
        px_lon = pixel_size_m / (111_320.0 * math.cos(math.radians(mid_lat)))

    log.info(f'    linhas={len(row_lats)}  colunas={len(col_lons)}  '
             f'px_lat={px_lat:.8f}°  px_lon={px_lon:.8f}°  patches={len(patches)}')

    # Corner superior-esquerdo do mosaico (a partir dos centros + meio patch)
    top_lat  = max(lats) + (PATCH_SIZE / 2) * px_lat
    left_lon = min(lons) - (PATCH_SIZE / 2) * px_lon

    bot_lat  = min(lats) - (PATCH_SIZE / 2) * px_lat
    right_lon= max(lons) + (PATCH_SIZE / 2) * px_lon

    total_h = max(1, round((top_lat  - bot_lat)  / px_lat))
    total_w = max(1, round((right_lon - left_lon) / px_lon))

    mosaic  = np.zeros((total_h, total_w), dtype=np.float32)
    weights = np.zeros((total_h, total_w), dtype=np.float32)

    for arr, meta in patches:
        lat, lon = meta['latitude'], meta['longitude']
        cy = round((top_lat - lat) / px_lat)   # linha do centro no mosaico
        cx = round((lon - left_lon) / px_lon)   # coluna do centro no mosaico
        y0, x0 = cy - PATCH_SIZE // 2, cx - PATCH_SIZE // 2
        y1, x1 = y0 + PATCH_SIZE,      x0 + PATCH_SIZE

        # Clamp aos limites do mosaico
        sy0, sy1 = max(0, y0), min(total_h, y1)
        sx0, sx1 = max(0, x0), min(total_w, x1)
        ay0, ay1 = sy0 - y0, sy1 - y0
        ax0, ax1 = sx0 - x0, sx1 - x0

        mosaic [sy0:sy1, sx0:sx1] += arr[ay0:ay1, ax0:ax1]
        weights[sy0:sy1, sx0:sx1] += 1.0

    weights = np.where(weights == 0, 1.0, weights)
    mosaic  = mosaic / weights

    affine = Affine(px_lon, 0.0, left_lon, 0.0, -px_lat, top_lat)
    return mosaic, affine, 'EPSG:4326'


def mosaic_patches(npy_files: list[Path],
                   pixel_size_m: float = 4.77) -> tuple[np.ndarray, Affine, str]:
    """Detecta formato (row/col ou lat/lon) e delega para o mosaicador correto."""
    patches = _load_patches(npy_files)
    if not patches:
        raise ValueError('Nenhum patch válido encontrado.')

    first = patches[0][1]
    if 'row' in first and 'col' in first:
        return _mosaic_rowcol(patches)
    elif 'latitude' in first and 'longitude' in first:
        return _mosaic_latlon(patches, pixel_size_m)
    else:
        raise ValueError(f'Formato de metadados não reconhecido. Chaves: {list(first.keys())}')


def save_tif(mosaic: np.ndarray, transform: Affine, crs_str: str, out_path: Path,
             threshold: float | None = None):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if threshold is not None:
        data  = (mosaic > threshold).astype(np.uint8)
        dtype = np.uint8
        log.info(f'    threshold={threshold}  positivos={data.sum():,}px')
    else:
        data  = mosaic
        dtype = np.float32

    with rasterio.open(
        out_path, 'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=dtype,
        crs=CRS.from_string(crs_str),
        transform=transform,
        compress='lzw',
    ) as dst:
        dst.write(data, 1)
    log.info(f'    salvo → {out_path.name}  ({data.shape[0]}×{data.shape[1]}px)')


def main():
    parser = argparse.ArgumentParser(
        description='Junta patches _pred.npy em GeoTIFF por região×ano')
    parser.add_argument('--predict-dir', type=Path, required=True,
                        help='Raiz das predições (<region>/<year>/patch_*_pred.npy)')
    parser.add_argument('--output-dir',  type=Path, required=True,
                        help='Pasta de saída dos GeoTIFF (estrutura plana)')
    parser.add_argument('--years',   type=int, nargs='+', default=None,
                        help='Anos a processar: dois valores = intervalo inclusivo '
                             '(ex: --years 2022 2025 → 2022..2025); '
                             'um ou mais de dois = lista explícita')
    parser.add_argument('--regions', type=str, nargs='+', default=None,
                        help='IDs de região a processar (padrão: todos)')
    parser.add_argument('--pixel-size', type=float, default=4.77,
                        help='Resolução em metros para patches lat/lon (padrão: 4.77 — Planet NICFI)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Limiar de binarização: pixels > threshold → 1 (padrão: 0.5). '
                             'Use --threshold 0 para salvar probabilidades float32 sem binarizar.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Reprocessa mesmo que o TIF de saída já exista.')
    args = parser.parse_args()

    if args.years and len(args.years) == 2:
        args.years = list(range(args.years[0], args.years[1] + 1))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for region_dir in sorted(args.predict_dir.iterdir()):
        if not region_dir.is_dir():
            continue
        if args.regions and region_dir.name not in args.regions:
            continue

        region_base = region_dir.name

        for year_dir in sorted(region_dir.iterdir()):
            if not year_dir.is_dir() or not year_dir.name.isdigit():
                continue
            year = int(year_dir.name)
            if args.years and year not in args.years:
                continue

            npy_files = sorted(year_dir.glob('patch_*_pred.npy'))
            if not npy_files:
                log.warning(f'  sem patches: {region_dir.name}/{year_dir.name}')
                continue

            out_path = args.output_dir / f'pred_{region_base}_{year}.tif'
            if out_path.exists() and not args.overwrite:
                log.info(f'  já existe, pulando: {out_path.name}')
                continue

            log.info(f'Processando  {region_dir.name}/{year_dir.name}  ({len(npy_files)} patches)')
            try:
                mosaic, transform, crs = mosaic_patches(npy_files, args.pixel_size)
                thr = args.threshold if args.threshold > 0 else None
                save_tif(mosaic, transform, crs, out_path, threshold=thr)
            except Exception as exc:
                log.error(f'  Erro em {region_dir.name}/{year_dir.name}: {exc}')

    log.info(f'Concluído. Log: {LOG_FILE.resolve()}')


if __name__ == '__main__':
    main()
