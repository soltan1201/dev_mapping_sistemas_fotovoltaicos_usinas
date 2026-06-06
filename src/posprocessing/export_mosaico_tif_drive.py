#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export Mosaico NICFI → GeoTIFF no Google Drive
===============================================
Exporta o mosaico Planet NICFI (5 bandas, Int16) recortado pela geometria
de cada região fotovoltaica para o Google Drive via ee.batch.Export.

Cada tarefa GEE gera:
  <DRIVE_FOLDER>/<system_index>_<year>.tif   →  GeoTIFF COG Int16

Bandas (5):
  0 blue  1 green  2 red  3 pvi  4 pvpi

Uso:
  python export_mosaico_tif_drive.py \
      --drive-folder fotovoltaica_mosaicos \
      [--years 2022 2024] \
      [--region-start 0] [--region-end 90] \
      [--max-active-tasks 50]
"""

import sys
import argparse
import time
import logging
import ee
from pathlib import Path

pathparent = str(Path(__file__).resolve().parents[1])
sys.path.append(pathparent)
from configure_account_projects_ee import get_current_account

# ==============================================================================
# 1. LOGGING
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ==============================================================================
# 2. ARGUMENTOS
# ==============================================================================
_ap = argparse.ArgumentParser(
    description='Exporta mosaico NICFI (5 bandas Int16) para Google Drive')
_ap.add_argument('--drive-folder', required=True,
                 help='Nome da pasta no Google Drive onde os TIFs serão salvos')
_ap.add_argument('--years', nargs='+', type=int,
                 default=[2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016],
                 help='Anos a exportar (ex: --years 2022 2024)')
_ap.add_argument('--region-start', type=int, default=0,
                 help='Índice inicial das regiões (inclusive, default=0)')
_ap.add_argument('--region-end', type=int, default=9999,
                 help='Índice final das regiões (exclusive, default=todas)')
_ap.add_argument('--max-active-tasks', type=int, default=50,
                 help='Máximo de tarefas GEE ativas em paralelo (default=50)')
_args = _ap.parse_args()

DRIVE_FOLDER     = _args.drive_folder
YEARS            = _args.years
REGION_INIC      = _args.region_start
REGION_END       = _args.region_end
MAX_ACTIVE_TASKS = _args.max_active_tasks

# ==============================================================================
# 3. CONFIGURAÇÕES
# ==============================================================================
ASSET_REGIONS = "projects/mapbiomas-arida/energias/shp_revisao2_16_05_2026_buffer_fotovoltaic_5km"
ASSET_NICFI   = 'projects/planet-nicfi/assets/basemaps/americas'

SCALE = 4.77          # metros/pixel — resolução nativa NICFI Planet
CRS   = 'EPSG:3857'   # projeção métrica; pixels quadrados e uniformes

NICFI_BANDS_SRC = ['B', 'G', 'R', 'N']
NICFI_BANDS_DST = ['blue', 'green', 'red', 'nir']

dict_percentil = {
    "blue":  [100,   800],
    "green": [300,  1200],
    "red":   [176,  1700],
    "nir":   [350,  4000],
}

# ==============================================================================
# 4. INICIALIZAÇÃO EE
# ==============================================================================
projAccount = get_current_account()
log.info(f"Projeto selecionado: {projAccount}")

try:
    ee.Initialize(project=projAccount)
    log.info('Earth Engine inicializado com sucesso.')
except Exception as e:
    log.error(f"Erro de inicialização: {e}")
    raise

# ==============================================================================
# 5. FUNÇÕES
# ==============================================================================

def build_nicfi_mosaic(year: int, geometry) -> ee.Image:
    """Mosaico mediana jul–dez do NICFI, 5 bandas normalizadas (Int16 [0,10000])."""
    mosaic = (ee.ImageCollection(ASSET_NICFI)
              .filterDate(f'{year}-07-01', f'{year + 1}-01-01')
              .filterBounds(geometry)
              .select(NICFI_BANDS_SRC, NICFI_BANDS_DST)
              .median()
              .toInt16())

    def normaliza(band_name):
        p_low  = ee.Number(dict_percentil[band_name][0])
        p_high = ee.Number(dict_percentil[band_name][1])
        return (mosaic.select(band_name)
                .subtract(p_low)
                .divide(p_high.subtract(p_low))
                .clamp(0, 1)
                .rename(band_name))

    norm_bands = [normaliza(b) for b in NICFI_BANDS_DST]
    scaled = norm_bands[0]
    for img in norm_bands[1:]:
        scaled = scaled.addBands(img)

    pvi = (mosaic.expression(
               'float(BLUE - NIR) / float(BLUE + NIR + 1)',
               {'BLUE': mosaic.select('blue'), 'NIR': mosaic.select('nir')})
           .add(1).divide(2).multiply(10000).toInt16().rename('pvi'))

    pvpi = (mosaic.expression(
                'float((green - blue) / (green + blue))',
                {'green': mosaic.select('green'), 'blue': mosaic.select('blue')})
            .add(1).divide(2).multiply(10000).toInt16().rename('pvpi'))

    return (scaled.multiply(10000).toInt16().select(['blue', 'green', 'red'])
            .addBands(pvi)
            .addBands(pvpi))


def count_active_tasks() -> int:
    """Número de tarefas GEE no estado RUNNING ou READY."""
    tasks = ee.data.getTaskList()
    return sum(1 for t in tasks if t['state'] in ('RUNNING', 'READY'))


def wait_for_task_slot(max_active: int, poll_s: int = 30):
    """Bloqueia até existirem menos de max_active tarefas ativas."""
    while True:
        active = count_active_tasks()
        if active < max_active:
            return
        log.info(f"  {active} tarefas ativas — aguardando {poll_s}s...")
        time.sleep(poll_s)


# ==============================================================================
# 6. PIPELINE PRINCIPAL
# ==============================================================================
log.info("Carregando feature collection de regiões fotovoltaicas...")
regions_fc    = ee.FeatureCollection(ASSET_REGIONS)
region_list   = regions_fc.toList(regions_fc.size())
total_regions = regions_fc.size().getInfo()

region_end_eff = min(REGION_END, total_regions)
n_tasks = (region_end_eff - REGION_INIC) * len(YEARS)

log.info(f"Total de regiões: {total_regions} | "
         f"Processando [{REGION_INIC}:{region_end_eff}] × {len(YEARS)} anos = {n_tasks} tarefas")
log.info(f"Destino: Google Drive / {DRIVE_FOLDER}")

submitted = 0

for cc in range(region_end_eff - REGION_INIC):
    global_idx = REGION_INIC + cc
    feature    = ee.Feature(region_list.get(global_idx))
    geom       = feature.geometry()

    feat_id      = feature.get('system:index').getInfo() or f'{global_idx:04d}'
    feat_id_safe = str(feat_id).replace('/', '_').replace(':', '_')

    log.info(f"\n{'='*60}")
    log.info(f"[{global_idx + 1}/{total_regions}] Região: {feat_id_safe}")

    for year in YEARS:
        task_name = f"{feat_id_safe}_{year}"

        wait_for_task_slot(MAX_ACTIVE_TASKS)

        image = build_nicfi_mosaic(year, geom)

        task = ee.batch.Export.image.toDrive(
            image          = image,
            description    = task_name,
            folder         = DRIVE_FOLDER,
            fileNamePrefix = task_name,
            region         = geom,
            scale          = SCALE,
            crs            = CRS,
            maxPixels      = 1e10,
            fileFormat     = 'GeoTIFF',
            formatOptions  = {'cloudOptimized': True},
        )
        task.start()
        submitted += 1
        log.info(f"  ✓ Submetida [{submitted}/{n_tasks}]: {task_name}")

        time.sleep(0.5)  # evita burst na API do GEE

log.info(f"\n{'='*60}")
log.info(f"Concluído. {submitted} tarefas submetidas.")
log.info("Monitore em: https://code.earthengine.google.com/tasks")
