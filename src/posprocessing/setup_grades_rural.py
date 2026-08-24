#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
setup_grades_rural.py
======================
Passo 0 — rodar UMA vez (ou quando quiser atualizar o universo de grades).

O que faz:
  1. Busca no GEE todas as grades que intersectam fotovoltaic_rural e conta
     quantos pontos FV cada uma tem.
  2. Exporta essa FeatureCollection como asset GEE:
       projects/mapbiomas-arida/energias/shp_grades_BR_fotovoltaic_rural
  3. Salva localmente:
       src/dados/log_rurais/grades_rural_all.json
     com formato: [{"system_index": "...", "quantidade": N}, ...]
     ordenado decrescente por quantidade.
  4. Encerra.

Após concluído (e o asset GEE exportado), execute o script principal:
  python export_mosaicTIF_GEEtoDrive_rural.py --drive-folder ... --year ...

Uso:
  python setup_grades_rural.py
  python setup_grades_rural.py --wait   # aguarda o export do asset
"""

import sys
import json
import argparse
import time
import logging
from pathlib import Path

import ee

pathparent = str(Path(__file__).resolve().parents[1])
sys.path.append(pathparent)
from configure_account_projects_ee import get_current_account

# ==============================================================================
# LOGGING
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ==============================================================================
# ARGUMENTOS
# ==============================================================================
_ap = argparse.ArgumentParser(
    description='Passo 0: exporta grades FV rural como asset GEE e salva JSON local')
_ap.add_argument('--wait', action='store_true',
                 help='Aguarda a conclusão do export do asset no GEE antes de encerrar')
_args = _ap.parse_args()

WAIT = _args.wait

# ==============================================================================
# CAMINHOS
# ==============================================================================
_SCRIPT_DIR = Path(__file__).resolve().parent
_LOG_DIR    = _SCRIPT_DIR.parent / 'dados' / 'log_rurais'
_LOG_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# ASSETS GEE
# ==============================================================================
ASSET_GRADE        = 'projects/nexgenmap/SAD_MapBiomas/DL/SHP_grades_BR_35pathces_AllBrV3'
ASSET_PHOTOVOLTAIC = 'projects/mapbiomas-arida/fotovoltaic_rural'
ASSET_OUTPUT       = 'projects/mapbiomas-arida/energias/shp_grades_BR_fotovoltaic_rural'
NAME_FC_OUTPUT     = 'shp_grades_BR_fotovoltaic_rural'

# ==============================================================================
# INICIALIZAÇÃO EE
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
# FUNÇÕES
# ==============================================================================

def build_grades_fc() -> tuple:
    """
    Constrói a FeatureCollection de grades com pontos FV e retorna:
      - grades_ee   : ee.FeatureCollection com propriedade 'quantidade'
      - grades_list : lista Python [{system_index, quantidade}, ...] ordenada desc
    """
    log.info('Calculando quantidade de pontos FV por grade (GEE)…')

    colection_fv = ee.FeatureCollection(ASSET_PHOTOVOLTAIC)
    grid_shp     = ee.FeatureCollection(ASSET_GRADE)
    grades       = grid_shp.filterBounds(colection_fv)

    def add_quantity(feat):
        quant = colection_fv.filterBounds(feat.geometry()).size()
        return feat.set('quantidade', quant)

    grades_ee = grades.map(add_quantity)

    log.info('Baixando lista de grades via getInfo()…')
    features = grades_ee.select(['quantidade']).getInfo()['features']

    grades_list = [
        {
            'system_index': feat['id'],
            'quantidade':   feat['properties'].get('quantidade', 0),
        }
        for feat in features
    ]
    grades_list.sort(key=lambda x: x['quantidade'], reverse=True)

    log.info(f'Total de grades com FV: {len(grades_list)}')
    return grades_ee, grades_list


def export_asset(grades_ee: ee.FeatureCollection) -> ee.batch.Task:
    """Submete Export.table.toAsset e retorna a task."""
    task = ee.batch.Export.table.toAsset(
        collection  = grades_ee,
        description = NAME_FC_OUTPUT,
        assetId     = ASSET_OUTPUT,
    )
    task.start()
    log.info(f'Export para asset iniciado: {ASSET_OUTPUT}')
    log.info('Acompanhe em: https://code.earthengine.google.com/tasks')
    return task


def wait_for_task(task: ee.batch.Task, poll_s: int = 30):
    """Aguarda a task de export até estado terminal."""
    log.info('Aguardando conclusão do export do asset…')
    while True:
        state = task.status()['state']
        log.info(f'  Estado: {state}')
        if state in ('COMPLETED', 'FAILED', 'CANCELLED'):
            if state != 'COMPLETED':
                log.error(f'Export encerrou com estado inesperado: {state}')
            return state
        time.sleep(poll_s)


def save_all_json(grades_list: list) -> Path:
    """Salva grades_rural_all.json em _LOG_DIR."""
    path = _LOG_DIR / 'grades_rural_all.json'
    path.write_text(
        json.dumps(grades_list, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    log.info(f'JSON salvo: {path}  ({len(grades_list)} grades)')
    return path


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    log.info('=' * 60)
    log.info(f'Asset out: {ASSET_OUTPUT}')
    log.info(f'Log dir  : {_LOG_DIR}')
    log.info(f'Aguardar : {WAIT}')
    log.info('=' * 60)

    grades_ee, grades_list = build_grades_fc()

    # Salva JSON imediatamente (não depende do export do asset)
    save_all_json(grades_list)

    # Exporta FeatureCollection como asset GEE
    task = export_asset(grades_ee)

    if WAIT:
        final_state = wait_for_task(task)
        log.info(f'Export finalizado com estado: {final_state}')
    else:
        log.info('Export submetido em background — não aguardando conclusão.')
        log.info('Aguarde o asset aparecer antes de rodar o script principal.')

    log.info('=' * 60)
    log.info('Setup concluído.')
    log.info('Próximo passo (após asset exportado):')
    log.info('  python export_mosaicTIF_GEEtoDrive_rural.py --drive-folder <pasta> --year <ano>')
    log.info('=' * 60)


if __name__ == '__main__':
    main()
