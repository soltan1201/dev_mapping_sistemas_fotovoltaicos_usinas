#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_mosaicTIF_GEEtoDrive_rural.py
=====================================
Exporta mosaico Planet NICFI (5 bandas Int16) recortado pelas grades do grid
BR que contêm pontos do asset fotovoltaic_rural.

PRÉ-REQUISITO: rodar setup_grades_rural.py --year <ano> uma vez antes.
Isso gera:
  - Asset GEE  : projects/mapbiomas-arida/energias/shp_grades_BR_fotovoltaic_rural
  - JSON local : src/dados/log_rurais/grades_rural_all_<year>.json

Assets GEE utilizados:
  grades (pré-filtradas): projects/mapbiomas-arida/energias/shp_grades_BR_fotovoltaic_rural
  nicfi                 : projects/planet-nicfi/assets/basemaps/americas

Controle de progresso:
  - Universo lido do JSON local grades_rural_all_<year>.json (sem chamar GEE).
  - Já submetidas identificadas pelos JSONs grades_rural_<year>_*.json em log_rurais/.
  - Pendentes = universo - já submetidas (comparação só entre JSONs locais).
  - Ao final do batch, salva grades_rural_<year>_<timestamp>.json com esta execução.

Uso:
  python export_mosaicTIF_GEEtoDrive_rural.py \\
      --drive-folder fotovoltaica_rural_mosaicos \\
      --year 2024 \\
      --limit 100 \\
      [--max-active-tasks 50]
"""

import sys
import json
import argparse
import time
import logging
from datetime import datetime
from pathlib import Path

import ee

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
    description='Exporta mosaico NICFI de grades rurais com FV para Google Drive')
_ap.add_argument('--drive-folder', required=True,
                 help='Pasta no Google Drive onde os TIFs serão salvos')
_ap.add_argument('--year', type=int, default=2024,
                 help='Ano do mosaico NICFI a exportar (padrão: 2024)')
_ap.add_argument('--limit', type=int, default=100,
                 help='Número máximo de grades a submeter neste batch (padrão: 100)')
_ap.add_argument('--max-active-tasks', type=int, default=500,
                 help='Máximo de tarefas GEE ativas em paralelo (padrão: 500)')
_args = _ap.parse_args()

DRIVE_FOLDER     = _args.drive_folder
YEAR             = _args.year
LIMIT            = _args.limit
MAX_ACTIVE_TASKS = _args.max_active_tasks

# ==============================================================================
# 3. CAMINHOS
# ==============================================================================
_SCRIPT_DIR  = Path(__file__).resolve().parent
_LOG_DIR     = _SCRIPT_DIR.parent / 'dados' / 'log_rurais'
_LOG_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 4. CONFIGURAÇÕES GEE
# ==============================================================================
ASSET_GRADE_FV = 'projects/mapbiomas-arida/energias/shp_grades_BR_fotovoltaic_rural'
ASSET_NICFI    = 'projects/planet-nicfi/assets/basemaps/americas'

SCALE = 4.77
CRS   = 'EPSG:3857'

NICFI_BANDS_SRC = ['B', 'G', 'R', 'N']
NICFI_BANDS_DST = ['blue', 'green', 'red', 'nir']

dict_percentil = {
    'blue':  [100,   800],
    'green': [300,  1200],
    'red':   [176,  1700],
    'nir':   [350,  4000],
}

# ==============================================================================
# 5. INICIALIZAÇÃO EE
# ==============================================================================
projAccount = get_current_account()
log.info(f'Projeto GEE: {projAccount}')

try:
    ee.Initialize(project=projAccount)
    log.info('Earth Engine inicializado com sucesso.')
except Exception as e:
    log.error(f'Erro de inicialização: {e}')
    raise

# ==============================================================================
# 6. LEITURA DOS JSONS LOCAIS (sem chamar GEE)
# ==============================================================================

def load_all_grades() -> list:
    """
    Lê grades_rural_all.json — universo completo gerado pelo setup.
    Retorna lista de dicts [{'system_index': str, 'quantidade': int}, ...]
    já ordenada decrescente por quantidade.
    """
    path = _LOG_DIR / 'grades_rural_all.json'
    if not path.exists():
        log.error(f'Arquivo de universo não encontrado: {path}')
        log.error('Execute primeiro: python setup_grades_rural.py')
        sys.exit(1)
    entries = json.loads(path.read_text(encoding='utf-8'))
    log.info(f'Universo carregado: {len(entries)} grades — {path.name}')
    return entries


def load_submitted_ids(year: int) -> set:
    """
    Lê todos os JSONs grades_rural_<year>_*.json em _LOG_DIR e retorna
    o conjunto de system_index já submetidos para este ano.
    Ignora o arquivo de universo (grades_rural_all_<year>.json).
    """
    done: set = set()
    pattern   = f'grades_rural_{year}_*.json'
    log_files = sorted(_LOG_DIR.glob(pattern))

    if not log_files:
        log.info(f'Nenhum log de submissão anterior encontrado para o ano {year}.')
        return done

    for lf in log_files:
        try:
            entries = json.loads(lf.read_text(encoding='utf-8'))
            for entry in entries:
                sid = entry.get('system_index')
                if sid:
                    done.add(sid)
        except Exception as exc:
            log.warning(f'Erro ao ler log {lf.name}: {exc}')

    log.info(f'Grades já submetidas (ano {year}): {len(done)} — de {len(log_files)} arquivo(s)')
    return done


def load_pending_grades(year: int) -> list:
    """
    Retorna lista de dicts {system_index, quantidade} que ainda não foram
    submetidas, comparando apenas JSONs locais (sem chamar o GEE).
    """
    all_grades   = load_all_grades()
    already_done = load_submitted_ids(year)
    pending = [g for g in all_grades if g['system_index'] not in already_done]
    log.info(f'Grades pendentes: {len(pending)} de {len(all_grades)} total')
    return pending


# ==============================================================================
# 7. FUNÇÕES DE EXPORTAÇÃO
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
    tasks = ee.data.getTaskList()
    return sum(1 for t in tasks if t['state'] in ('RUNNING', 'READY'))


def wait_for_task_slot(max_active: int, poll_s: int = 30):
    while True:
        active = count_active_tasks()
        if active < max_active:
            return
        log.info(f'  {active} tarefas ativas — aguardando {poll_s}s…')
        time.sleep(poll_s)


def submit_task(region_id: str, geom, year: int, submitted: int, n_tasks: int):
    task_name = f'{region_id}_{year}'
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
    log.info(f'  ✓ Submetida [{submitted}/{n_tasks}]: {task_name}')
    time.sleep(0.5)


# ==============================================================================
# 8. PIPELINE PRINCIPAL
# ==============================================================================

def save_log(submitted_entries: list, year: int):
    """Salva JSON com as grades submetidas nesta execução."""
    today    = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = _LOG_DIR / f'grades_rural_{year}_{today}.json'
    log_path.write_text(
        json.dumps(submitted_entries, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    log.info(f'Log salvo: {log_path}')


def main():
    log.info('=' * 60)
    log.info(f'Ano        : {YEAR}')
    log.info(f'Limit      : {LIMIT} grades')
    log.info(f'Drive      : {DRIVE_FOLDER}')
    log.info(f'Log dir    : {_LOG_DIR}')
    log.info(f'Asset grade: {ASSET_GRADE_FV}')
    log.info('=' * 60)

    # Pendentes = universo (JSON local) - já submetidas (logs locais); sem chamar GEE
    pending = load_pending_grades(YEAR)

    batch = pending[:LIMIT]
    if not batch:
        log.info('Nenhuma grade nova para submeter. Pipeline concluído.')
        return

    n_tasks  = len(batch)
    log.info(f'Submetendo {n_tasks} grade(s) — ano {YEAR}')
    log.info('=' * 60)

    grid_fv = ee.FeatureCollection(ASSET_GRADE_FV)

    submitted_entries = []
    submitted_count   = 0

    for grade in batch:
        region_id  = grade['system_index']
        quantidade = grade['quantidade']

        feature = grid_fv.filter(ee.Filter.eq('system:index', region_id)).first()
        geom    = feature.geometry()

        submitted_count += 1
        log.info(f'\n{"="*60}')
        log.info(f'Grade: {region_id}  qtd_fv={quantidade}  ({submitted_count}/{n_tasks})')

        try:
            submit_task(region_id, geom, YEAR, submitted_count, n_tasks)
            submitted_entries.append({
                'system_index': region_id,
                'quantidade':   quantidade,
                'year':         YEAR,
                'task_name':    f'{region_id}_{YEAR}',
            })
        except Exception as exc:
            log.error(f'  Erro ao submeter {region_id}: {exc}')

    log.info(f'\n{"="*60}')
    log.info(f'Concluído. {submitted_count} tarefa(s) submetida(s).')
    log.info('Monitore em: https://code.earthengine.google.com/tasks')

    if submitted_entries:
        save_log(submitted_entries, YEAR)


if __name__ == '__main__':
    main()
