#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exporta mosaico anual de FV — Coleção 11 → asset público
=========================================================
Tarefa 1: Junta todas as imagens da ImageCollection fonte por ano e exporta
          um mosaico por ano (todos os tiles/regiões fundidos).

Tarefa 2: Detecta qual região de 2018 está faltando (todos os outros anos têm
          94 imagens, 2018 tem 93). Preenche a lacuna copiando o dado de 2019
          para essa região antes de montar o mosaico de 2018.
          Suspeita: 00000000000000000007

Tarefa 3: Reprojeta o mosaico para 30 m (Landsat) antes de exportar.

Fonte : projects/mapbiomas-workspace/AMOSTRAS/col11/CAATINGA/layers_energia
Saída : projects/mapbiomas-brazil/assets/LAND-COVER/COLLECTION-11/SOLAR-PANELS/classification
Nome  : solar-panel-{year}-{VERSION}
"""

import sys
import os
from pathlib import Path
from collections import defaultdict
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

VERSION_IN  = 2          # versão dos tiles na coleção fonte
VERSION_OUT = 2          # versão do mosaico exportado
SCALE_OUT   = 30         # m/pixel — Landsat (saída pública)
CRS         = 'EPSG:4326'
YEARS       = list(range(2016, 2026))   # 2016 → 2025
SKIP_EXISTING = True

SUSPECT_MISSING_2018 = '0000000000000000000e'   # tile ausente em 2018 (confirmado)

limite_brasil = ee.FeatureCollection(ASSET_LIMITE_BRASIL).geometry()

# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================

def asset_exists(asset_id):
    try:
        ee.data.getAsset(asset_id)
        return True
    except Exception:
        return False


def list_collection_assets(collection_id):
    """Lista todos os assets de uma ImageCollection com paginação."""
    assets = []
    params = {'parent': collection_id}
    while True:
        resp = ee.data.listAssets(params)
        assets.extend(resp.get('assets', []))
        next_token = resp.get('nextPageToken')
        if not next_token:
            break
        params['pageToken'] = next_token
    return assets


def parse_asset_name(full_name):
    """
    Extrai (year, region_id, version) do nome completo do asset.
    Formato: .../solar-panel-{year}-{region_id}-{version}
    """
    basename = full_name.split('/')[-1]         # solar-panel-2018-000...0007-1
    parts    = basename.split('-')              # ['solar','panel','2018','000...0007','1']
    if len(parts) < 5:
        return None, None, None
    try:
        year      = int(parts[2])
        region_id = parts[3]
        version   = int(parts[4])
        return year, region_id, version
    except (ValueError, IndexError):
        return None, None, None


# ==============================================================================
# TAREFA 2 — Detecta região ausente em 2018
# ==============================================================================

print("=" * 60)
print("TAREFA 2 — Detectando região ausente em 2018")
print("=" * 60)
print("Listando assets na coleção fonte (pode demorar)...")

all_assets = list_collection_assets(ASSET_SOURCE)
print(f"Total de assets encontrados: {len(all_assets)}\n")

# Agrupa region_ids por ano (apenas VERSION_IN para evitar contar v2 duplicado)
regions_by_year = defaultdict(set)

for asset in all_assets:
    name = asset.get('name', asset.get('id', ''))
    year, region_id, version = parse_asset_name(name)
    if year in YEARS and region_id and version == VERSION_IN:
        regions_by_year[year].add(region_id)

print("Contagem de regiões por ano (versão %d):" % VERSION_IN)
for y in YEARS:
    print(f"  {y}: {len(regions_by_year[y])} regiões")

# Conjunto de referência: todos os region_ids encontrados em qualquer ano
all_region_ids = set()
for y in YEARS:
    all_region_ids |= regions_by_year[y]

missing_2018 = all_region_ids - regions_by_year[2018]

if missing_2018:
    print(f"\nRegiões ausentes em 2018: {sorted(missing_2018)}")
else:
    # Nenhuma detectada pela listagem — usa a suspeita como fallback
    missing_2018 = {SUSPECT_MISSING_2018}
    print(f"\nNenhuma região ausente detectada pela listagem.")
    print(f"Aplicando suspeita: {SUSPECT_MISSING_2018}")

FILL_YEAR = 2019    # ano usado para preencher a lacuna de 2018

print(f"\nAs regiões ausentes em 2018 serão preenchidas com dados de {FILL_YEAR}.")

# ==============================================================================
# TAREFA 1 + 3 — Mosaico anual + reprojeção para 30 m + exportação
# ==============================================================================

def build_annual_mosaic(year, missing_regions):
    """
    Carrega todas as imagens do ano na coleção fonte, preenche lacunas
    (2018) com dados do FILL_YEAR, mosaica e reprojeta para SCALE_OUT.
    """
    col = (
        ee.ImageCollection(ASSET_SOURCE)
        .filter(ee.Filter.eq('year', year))
        .filter(ee.Filter.eq('version', VERSION_IN))
        .select('b1')
    )

    # Preenche regiões ausentes em 2018 com tiles de 2019
    if missing_regions and year == 2018:
        fill_images = []
        for region_id in sorted(missing_regions):
            fill_asset = f'{ASSET_SOURCE}/solar-panel-{FILL_YEAR}-{region_id}-{VERSION_IN}'
            print(f"    Preenchendo região {region_id} com dados de {FILL_YEAR}")
            fill_img = ee.Image(fill_asset).select('b1').selfMask()
            fill_images.append(fill_img)
        if fill_images:
            col = col.merge(ee.ImageCollection(fill_images))

    return (
        col
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


print("\n" + "=" * 60)
print("TAREFA 1+3 — Mosaicos anuais (escala %d m)" % SCALE_OUT)
print(f"Saída: {ASSET_OUTPUT}")
print("=" * 60)

# sys.exit()  # descomente para testar sem disparar tasks

for year in YEARS:
    print(f"\n[{year}]")
    missing = missing_2018 if year == 2018 else set()
    img = build_annual_mosaic(year, missing)
    export_year(year, img)

print("\n" + "=" * 60)
print("Todas as exportações foram disparadas.")
print("Acompanhe as tasks no Earth Engine Code Editor → Tasks.")
