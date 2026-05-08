#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download Predict TFRecord - Fotovoltaica (Planet NICFI)
=======================================================
Exporta patches 257×257×8 bandas como TFRecord no Google Drive via
Export.table.toDrive() + neighborhoodToArray, usando tasks GEE em paralelo.

Vantagem sobre download_dataset_predict_tfrecord_fotovoltaica.py (.npy):
  • Múltiplos patches por task, sem limite de 1 req/s do computePixels().
  • O GEE processa tudo em paralelo nos servidores — muito mais rápido.

Pré-processamento:
  Idêntico ao dataset de treino (download_dataset_tfrecord_fotovoltaica.py):
  mesmos percentis e filtro jul–dez, garantindo consistência treino/predict.

Formato do TFRecord gerado (por Example):
  Cada feature = lista flat Int64 de 257×257 = 66 049 valores:
    'blue', 'green', 'red', 'nir', 'pvi', 'iia', 'ri', 'evi'
  Metadados adicionais (string / int):
    'region_id', 'year', 'latitude', 'longitude'

Uso:
  python download_predict_as_tfrecord_fv.py \\
      --year_inic 2016 --year_end 2022 \\
      --region_inic 0  --region_end 50 \\
      --drive-folder DS_FV_PREDICT_TFRECORDS
"""

import sys
import os
import math
import argparse
import collections
import ee
from pathlib import Path

collections.Callable = collections.abc.Callable

pathparent = str(Path(os.getcwd()).parents[0])
sys.path.append(pathparent)

try:
    ee.Initialize(project='mapbiomas-caatinga-cloud04')
    print('Earth Engine inicializado com sucesso!')
except ee.EEException as e:
    print('Falha ao inicializar Earth Engine:', e)
    raise
except Exception:
    print('Erro inesperado:', sys.exc_info()[0])
    raise

# ==============================================================================
# 1. CONFIGURAÇÕES
# ==============================================================================

ASSET_REGIONS = 'projects/mapbiomas-arida/update_02_05_2026_buffer_fotovoltaic_5km'
ASSET_NICFI   = 'projects/planet-nicfi/assets/basemaps/americas'

PATCH_SIZE        = 256   # pixels; kernel rect(128) → saída 257×257
SCALE             = 4.77  # m/pixel (resolução nativa NICFI Planet)
STRIDE_PIXELS     = 230   # espaçamento da grade em pixels
MAX_PATCHES_FILE  = 50    # máx patches por shard TFRecord

# Percentis idênticos ao dataset de treino
dict_percentil = {
    "blue":  [100,   800],
    "green": [300,  1200],
    "red":   [176,  1700],
    "nir":   [350,  4000],
}

NICFI_BANDS_SRC = ['B', 'G', 'R', 'N']
NICFI_BANDS_DST = ['blue', 'green', 'red', 'nir']
ALL_BANDS       = ['blue', 'green', 'red', 'nir', 'pvi', 'iia', 'ri', 'evi']
SELECTORS       = ALL_BANDS + ['region_id', 'year', 'latitude', 'longitude']

# ==============================================================================
# 2. FUNÇÕES — IMAGEM
# ==============================================================================

def build_nicfi_mosaic(year: int, geometry) -> ee.Image:
    """
    Mosaico mediana jul–dez do NICFI com 8 bandas em [0, 10000] Int16.
    Pré-processamento idêntico ao dataset de treino (filtro jul–dez,
    mesmos percentis por banda).
    """
    mosaic = (ee.ImageCollection(ASSET_NICFI)
              .filterDate(f'{year}-07-01', f'{year + 1}-01-01')
              .filterBounds(geometry)
              .select(NICFI_BANDS_SRC, NICFI_BANDS_DST)
              .median()
              .toInt16())

    def normaliza_banda(band_name):
        p_low  = ee.Number(dict_percentil[band_name][0])
        p_high = ee.Number(dict_percentil[band_name][1])
        return (mosaic.select(band_name)
                .subtract(p_low)
                .divide(p_high.subtract(p_low))
                .clamp(0, 1)
                .rename(band_name))

    norm_bands = [normaliza_banda(b) for b in NICFI_BANDS_DST]
    scaled_img = norm_bands[0]
    for img in norm_bands[1:]:
        scaled_img = scaled_img.addBands(img)

    pvi = (mosaic.expression(
               'float(BLUE - NIR) / float(BLUE + NIR + 1)',
               {'BLUE': mosaic.select('blue'), 'NIR': mosaic.select('nir')})
           .add(1).divide(2).multiply(10000).toInt16().rename('pvi'))

    iia = (mosaic.expression(
               "float((green - 4 * nir) / (green + 4 * nir + 1))",
               {'green': mosaic.select('green'), 'nir': mosaic.select('nir')})
           .add(1).divide(2).multiply(10000).toInt16().rename('iia'))

    ri = (mosaic.expression(
              "float(2.4 * (red - green) / (red + green + 1))",
              {'red': mosaic.select('red'), 'green': mosaic.select('green')})
          .add(2.4).divide(4.8).multiply(10000).toInt16().rename('ri'))

    evi = (mosaic.expression(
               "float(2.4 * (nir - red) / (1 + nir + red))",
               {'nir': mosaic.select('nir'), 'red': mosaic.select('red')})
           .add(2.4).divide(4.8).multiply(10000).toInt16().rename('evi'))

    return (scaled_img.multiply(10000).toInt16()
            .addBands(pvi).addBands(iia).addBands(ri).addBands(evi))


def build_patches_array(year: int, geometry) -> ee.Image:
    """Stack de 8 bandas com neighborhoodToArray — pronto para sampleRegions."""
    mosaic = build_nicfi_mosaic(year, geometry)
    kernel = ee.Kernel.rectangle(PATCH_SIZE // 2, PATCH_SIZE // 2, 'pixels')
    return (mosaic.select(ALL_BANDS)
            .unmask(0)
            .toInt16()
            .neighborhoodToArray(kernel))

# ==============================================================================
# 3. FUNÇÕES — GRADE E METADADOS
# ==============================================================================

def generate_grid_points(geometry) -> ee.FeatureCollection:
    """Grade regular de pontos em EPSG:3857 cobrindo a geometria."""
    proj      = ee.Projection('EPSG:3857').atScale(SCALE)
    px_coords = ee.Image.pixelCoordinates(proj)
    x_idx     = px_coords.select('x').divide(SCALE).round().toInt()
    y_idx     = px_coords.select('y').divide(SCALE).round().toInt()
    grid_mask = x_idx.mod(STRIDE_PIXELS).eq(0).And(y_idx.mod(STRIDE_PIXELS).eq(0))

    return (grid_mask.selfMask()
            .sample(region=geometry, scale=SCALE, projection=proj,
                    geometries=True, tileScale=4))


def tag_points(fc: ee.FeatureCollection,
               region_id: str, year: int) -> ee.FeatureCollection:
    """Adiciona region_id, year, latitude e longitude a cada ponto."""
    return fc.map(lambda f: f.set({
        'region_id': region_id,
        'year':      year,
        'longitude': f.geometry().coordinates().get(0),
        'latitude':  f.geometry().coordinates().get(1),
    }))

# ==============================================================================
# 4. FUNÇÕES — EXPORT
# ==============================================================================

def export_shard(patches_array: ee.Image,
                 points: ee.FeatureCollection,
                 region_id: str, year: int,
                 shard_idx: int,
                 drive_folder: str) -> str:
    """Envia uma task de export TFRecord para o Drive. Retorna o nome."""
    fname = f"predict_fv_{region_id}_{year}_part{shard_idx:03d}"
    task  = ee.batch.Export.table.toDrive(
        collection=patches_array.sampleRegions(
            collection=points,
            scale=SCALE,
            geometries=False,
            tileScale=16,
        ),
        description=fname,
        folder=drive_folder,
        fileFormat='TFRecord',
        selectors=SELECTORS,
    )
    task.start()
    return fname

# ==============================================================================
# 5. PIPELINE PRINCIPAL
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Exporta patches NICFI como TFRecord para o Drive (predict)')
    parser.add_argument('--year_inic',    type=int, default=2016,
                        help='Ano inicial (padrão: 2016)')
    parser.add_argument('--year_end',     type=int, default=2025,
                        help='Ano final, inclusivo (padrão: 2025)')
    parser.add_argument('--region_inic',  type=int, default=0,
                        help='Índice inicial de região (padrão: 0)')
    parser.add_argument('--region_end',   type=int, default=50,
                        help='Índice final de região, inclusivo (padrão: 50)')
    parser.add_argument('--drive-folder', type=str,
                        default='DS_FV_PREDICT_TFRECORDS',
                        help='Pasta no Google Drive (padrão: DS_FV_PREDICT_TFRECORDS)')
    args = parser.parse_args()

    years        = list(range(args.year_inic, args.year_end + 1))
    drive_folder = args.drive_folder

    print(f"\nAnos          : {years}")
    print(f"Regiões       : [{args.region_inic}, {args.region_end}]")
    print(f"Drive folder  : {drive_folder}")
    print(f"Asset regiões : {ASSET_REGIONS}\n")

    regions_fc    = ee.FeatureCollection(ASSET_REGIONS)
    region_list   = regions_fc.toList(regions_fc.size())
    total_regions = regions_fc.size().getInfo()
    region_end    = min(total_regions - 1, args.region_end)

    print(f"Total de regiões no asset : {total_regions}")
    print(f"Intervalo processado      : [{args.region_inic}, {region_end}] "
          f"({region_end - args.region_inic + 1} regiões)\n")

    tasks_started = 0

    for global_idx in range(args.region_inic, region_end + 1):
        feature      = ee.Feature(region_list.get(global_idx))
        geom         = feature.geometry()
        feat_id      = feature.get('system:index').getInfo() or f'{global_idx:04d}'
        feat_id_safe = str(feat_id).replace('/', '_').replace(':', '_')

        print(f"{'=' * 60}")
        print(f"[{global_idx + 1}/{total_regions}] Região: {feat_id_safe}")

        grid_points = generate_grid_points(geom)
        n_points    = grid_points.size().getInfo()
        print(f"  Grade: {n_points} pontos "
              f"(stride={STRIDE_PIXELS}px ≈ {STRIDE_PIXELS * SCALE:.0f} m)")

        if n_points == 0:
            print("  Região sem pontos na grade. Pulando.")
            continue

        for year in years:
            print(f"\n  --- Ano {year} ---")

            num_nicfi = (ee.ImageCollection(ASSET_NICFI)
                         .filterDate(f'{year}-07-01', f'{year + 1}-01-01')
                         .filterBounds(geom)
                         .size().getInfo())
            if num_nicfi == 0:
                print(f"  Sem imagens NICFI para {year}. Pulando.")
                continue

            patches_array = build_patches_array(year, geom)
            num_shards    = max(1, math.ceil(n_points / MAX_PATCHES_FILE))

            # Distribui pontos em shards + injeta metadados
            tagged_points = tag_points(grid_points, feat_id_safe, year)
            sample_points = tagged_points.randomColumn('shard_idx', seed=year)

            print(f"  Exportando {num_shards} shard(s) "
                  f"(máx {MAX_PATCHES_FILE} patches/shard)...")

            for i in range(num_shards):
                lo = i / num_shards
                hi = (i + 1) / num_shards

                shard_pts = (sample_points
                             .filter(ee.Filter.And(
                                 ee.Filter.gte('shard_idx', lo),
                                 ee.Filter.lt('shard_idx', hi)))
                             .limit(MAX_PATCHES_FILE))

                n_shard = shard_pts.size().getInfo()
                if n_shard == 0:
                    print(f"    Shard {i:03d}: vazio, pulando.")
                    continue

                try:
                    fname = export_shard(
                        patches_array, shard_pts,
                        feat_id_safe, year, i, drive_folder,
                    )
                    print(f"    Shard {i:03d}: {n_shard} pts → task '{fname}' enviada")
                    tasks_started += 1
                except Exception as exc:
                    print(f"    Erro shard {i:03d}: {exc}")

    print(f"\n{'=' * 60}")
    print(f"Concluído. {tasks_started} task(s) enviadas para o GEE.")
    print("Acompanhe em: https://code.earthengine.google.com/tasks")


if __name__ == '__main__':
    main()
