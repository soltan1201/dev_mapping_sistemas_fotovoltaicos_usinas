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


def infer_stride(patches: list[tuple]) -> int:
    """
    Calcula o stride em pixels a partir dos transforms de dois patches vizinhos.
    Usa col=0→1 (mesma linha) ou row=0→1 (mesma coluna) como referência.
    Fallback: PATCH_SIZE (sem sobreposição).
    """
    def get_meta(row, col):
        return next((m for _, m in patches if m['row'] == row and m['col'] == col), None)

    m00 = get_meta(0, 0)
    if m00 is None:
        return PATCH_SIZE

    m01 = get_meta(0, 1)
    if m01:
        scale_x = m00['transform'][0]
        return round((m01['transform'][2] - m00['transform'][2]) / scale_x)

    m10 = get_meta(1, 0)
    if m10:
        scale_y = abs(m00['transform'][4])
        return round(abs(m10['transform'][5] - m00['transform'][5]) / scale_y)

    return PATCH_SIZE


def mosaic_patches(npy_files: list[Path]) -> tuple[np.ndarray, Affine, str]:
    """
    Junta patches _pred.npy em um mosaico usando row/col do JSON.
    Regiões sobrepostas recebem média ponderada (blending).
    Retorna (mosaic_hw float32, Affine, crs_str).
    """
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

    if not patches:
        raise ValueError('Nenhum patch válido encontrado.')

    max_row = max(m['row'] for _, m in patches)
    max_col = max(m['col'] for _, m in patches)
    stride  = infer_stride(patches)

    log.info(f'    stride={stride}px  grid {max_row+1}×{max_col+1}  patches={len(patches)}')

    h = max_row * stride + PATCH_SIZE
    w = max_col * stride + PATCH_SIZE

    mosaic  = np.zeros((h, w), dtype=np.float32)
    weights = np.zeros((h, w), dtype=np.float32)

    for arr, meta in patches:
        r, c = meta['row'], meta['col']
        y0 = r * stride
        x0 = c * stride
        mosaic [y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE] += arr
        weights[y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE] += 1.0

    weights = np.where(weights == 0, 1.0, weights)
    mosaic  = mosaic / weights

    # Transform do patch (row=0, col=0) é a origem do mosaico
    t      = next(m for _, m in patches if m['row'] == 0 and m['col'] == 0)['transform']
    affine = Affine(t[0], t[1], t[2], t[3], t[4], t[5])
    crs    = next(m for _, m in patches if m['row'] == 0 and m['col'] == 0).get('crs', 'EPSG:4326')

    return mosaic, affine, crs


def save_tif(mosaic: np.ndarray, transform: Affine, crs_str: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out_path, 'w',
        driver='GTiff',
        height=mosaic.shape[0],
        width=mosaic.shape[1],
        count=1,
        dtype=np.float32,
        crs=CRS.from_string(crs_str),
        transform=transform,
        compress='lzw',
    ) as dst:
        dst.write(mosaic, 1)
    log.info(f'    salvo → {out_path.name}  ({mosaic.shape[0]}×{mosaic.shape[1]}px)')


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
            if out_path.exists():
                log.info(f'  já existe, pulando: {out_path.name}')
                continue

            log.info(f'Processando  {region_dir.name}/{year_dir.name}  ({len(npy_files)} patches)')
            try:
                mosaic, transform, crs = mosaic_patches(npy_files)
                save_tif(mosaic, transform, crs, out_path)
            except Exception as exc:
                log.error(f'  Erro em {region_dir.name}/{year_dir.name}: {exc}')

    log.info(f'Concluído. Log: {LOG_FILE.resolve()}')


if __name__ == '__main__':
    main()
