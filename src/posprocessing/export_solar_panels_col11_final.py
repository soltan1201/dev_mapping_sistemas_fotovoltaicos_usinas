#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exporta mosaico anual de FV — Coleção 11 (versão 2) → asset público
=====================================================================
Carrega os tiles v2 (já corrigidos) da coleção fonte, mosaica por ano,
reprojeta para 30 m e exporta um asset por ano.

Fonte : projects/mapbiomas-workspace/AMOSTRAS/col11/CAATINGA/layers_energia  (v2)
Saída : projects/mapbiomas-brazil/assets/LAND-COVER/COLLECTION-11/RENEWABLE-ENERGY/solar-panels
Nome  : solar-panel-{year}-{VERSION_OUT}
"""

import sys
import os
from pathlib import Path
import ee

pathparent = str(Path(os.getcwd()).parents[0])
sys.path.append(pathparent)
from configure_account_projects_ee import get_current_account

projAccount = get_current_account()
print(f"Projeto selecionado >>> {projAccount} <<<")

try:
    ee.Initialize(project=projAccount)
    print("Earth Engine inicializado com sucesso!\n")
except Exception as e:
    print("Erro de Inicialização:", e)
    raise

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================

ASSET_SOURCE        = 'projects/mapbiomas-workspace/AMOSTRAS/col11/CAATINGA/layers_energia'
ASSET_OUTPUT        = 'projects/mapbiomas-brazil/assets/LAND-COVER/COLLECTION-11/RENEWABLE-ENERGY/solar-panels'
ASSET_LIMITE_BRASIL = 'users/CartasSol/shapes/Brasil_Manual'

VERSION_IN    = 2          # tiles v2 já possuem todas as correções aplicadas
VERSION_OUT   = 2
SCALE_OUT     = 30         # m/pixel — Landsat (saída pública)
CRS           = 'EPSG:4326'
YEARS         = list(range(2016, 2026))   # 2016–2025
SKIP_EXISTING = True

limite_brasil = ee.FeatureCollection(ASSET_LIMITE_BRASIL).geometry()

# ==============================================================================
# FUNÇÕES
# ==============================================================================

def asset_exists(asset_id):
    try:
        ee.data.getAsset(asset_id)
        return True
    except Exception:
        return False


def build_annual_mosaic(year):
    return (
        ee.ImageCollection(ASSET_SOURCE)
        .filter(ee.Filter.eq('year', year))
        .filter(ee.Filter.eq('version', VERSION_IN))
        .select('b1')
        .mosaic()
        .selfMask()
        .byte()
        .rename('b1')
        .reproject(crs=CRS, scale=SCALE_OUT)
    )


def export_year(year, img):
    asset_name = f'solar-panel-{year}-{VERSION_OUT}'
    asset_id   = f'{ASSET_OUTPUT}/{asset_name}'

    if SKIP_EXISTING and asset_exists(asset_id):
        print(f'  [{year}] já existe — pulando.')
        return

    img_final = img.selfMask().set({
        'year':       year,
        'collection': 11,
        'theme':      'SOLAR-PANELS',
        'source':     'geodatin',
        'version':    VERSION_OUT,
        'territory':  'BRAZIL',
    })

    task = ee.batch.Export.image.toAsset(
        image=img_final,
        description=asset_name,
        assetId=asset_id,
        scale=SCALE_OUT,
        crs=CRS,
        region=limite_brasil,
        maxPixels=1e13,
        pyramidingPolicy={'b1': 'mode'},
    )
    task.start()
    print(f'  [{year}] task → {asset_id}')


# ==============================================================================
# PIPELINE
# ==============================================================================

print(f"Anos         : {YEARS}")
print(f"Versão tiles : {VERSION_IN}  →  mosaico: {VERSION_OUT}")
print(f"Escala saída : {SCALE_OUT} m")
print(f"Saída        : {ASSET_OUTPUT}")
print("=" * 60)

# sys.exit()  # descomente para inspecionar sem disparar tasks

for year in YEARS:
    print(f"\n[{year}]")
    img = build_annual_mosaic(year)
    export_year(year, img)

print("\n" + "=" * 60)
print("Todas as exportações foram disparadas.")
print("Acompanhe as tasks no Earth Engine Code Editor → Tasks.")
