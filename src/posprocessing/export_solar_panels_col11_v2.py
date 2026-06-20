#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pós-processamento Col11 — Versão 2
====================================
Aplica 3 conjuntos de regras sobre os tiles v1 e exporta como v2,
mantendo o mesmo formato: uma imagem por (region_id × year) na mesma collection.

Regra 1 — EXCLUIR
  Regiões zeradas em todos os anos (sem FV confirmado).

Regra 2 — MERGE COL10
  col10.add(col11_v1).gt(0)  por ano.
  Para 2025 do col11 usa col10 de 2024 (último ano disponível).

Regra 3 — ZERO BACK
  Zera todos os anos de 2016 até o ano X inclusive; anos > X mantêm v1
  (ou aplicam Regra 2 se a região também estiver nessa lista).
  Onde Regra 1 e Regra 3 se sobrepõem, Regra 1 tem prioridade.

Saída : mesmo ASSET_SOURCE  /  solar-panel-{year}-{region_id}-2
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

ASSET_SOURCE      = 'projects/mapbiomas-workspace/AMOSTRAS/col11/CAATINGA/layers_energia'
ASSET_COL10       = 'projects/mapbiomas-brazil/assets/LAND-COVER/COLLECTION-10/SOLAR-PANELS/classification'
ASSET_SHP_REGIONS = 'projects/mapbiomas-arida/energias/shp_revisao3_10_06_2026_buffer_fotovoltaic_5km'

VERSION_IN    = 1
VERSION_OUT   = 2
CRS           = 'EPSG:4326'
SCALE         = 4.77        # m/pixel — Planet NICFI
YEARS         = list(range(2016, 2026))   # 2016–2025
COL10_MAX_YEAR = 2024       # col10 não tem 2025; usa 2024 para o ano 2025 do col11
SKIP_EXISTING = True

# ==============================================================================
# REGRAS
# ==============================================================================

# Regra 1 — zera completamente em todos os anos
EXCLUDE_REGIONS = {
    '00000000000000000006',
    '00000000000000000007',
    '0000000000000000001d',
    '0000000000000000002b',
    '0000000000000000003a',
    '00000000000000000047',
    '0000000000000000004c',
    '00000000000000000058',
}

# Regra 2 — fusão col10.add(col11).gt(0) por ano
MERGE_COL10_REGIONS = {
    '0000000000000000000a',
    '0000000000000000000b',
    '0000000000000000000e',
    '00000000000000000026',
    '0000000000000000002a',
    '0000000000000000002e',
    '0000000000000000002f',
    '00000000000000000030',
    '00000000000000000032',
    '00000000000000000041',
    '00000000000000000042',
    '00000000000000000045',
    '0000000000000000005d',
    '00000000000000000061',
}

# Regra 3 — zero_back: zera de 2016 até o ano X inclusive (chave → último ano zerado)
ZERO_BACK_REGIONS = {
    '00000000000000000005': 2019,
    '0000000000000000000a': 2022,
    "0000000000000000001a": 2021,
    "0000000000000000001b": 2021,
    "0000000000000000000e": 2021,
    "0000000000000000000f": 2021,
    "00000000000000000020": 2020,
    "00000000000000000025": 2020,
    "00000000000000000025": 2019,
    '00000000000000000028': 2024,
    '0000000000000000002d': 2018,
    '00000000000000000038': 2020,
    '0000000000000000003a': 2022,   # também em EXCLUDE — EXCLUDE tem prioridade
    '0000000000000000003c': 2022,
    '0000000000000000003d': 2022,
    '0000000000000000003f': 2022,
    '00000000000000000040': 2022,
    '00000000000000000041': 2023,   # também em MERGE (anos > 2023 aplicam merge)
    '00000000000000000043': 2019,
    '0000000000000000004a': 2021,
    '0000000000000000004d': 2020,
    '0000000000000000004e': 2021,
    '0000000000000000004f': 2017,
    '00000000000000000050': 2016,
    '00000000000000000057': 2022,
}

# Assets v1 ausentes — (region_id, year_faltante): year_substituto
# O v1 do ano substituto é usado no lugar do v1 faltante.
MISSING_V1_FILL = {
    ('0000000000000000000e', 2018): 2019,   # 2018 não foi exportado; usa 2019
}

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
    """Extrai (year, region_id, version) de solar-panel-{year}-{region_id}-{version}."""
    basename = full_name.split('/')[-1]
    parts    = basename.split('-')
    if len(parts) < 5:
        return None, None, None
    try:
        return int(parts[2]), parts[3], int(parts[4])
    except (ValueError, IndexError):
        return None, None, None


def get_col10_image(year):
    """Carrega col10 para o ano (2025 → usa col10-2024).
    Banda na Col10: classification_{year} → renomeada para b1."""
    col10_year = min(year, COL10_MAX_YEAR)
    band_name  = f'classification_{col10_year}'
    return (
        ee.ImageCollection(ASSET_COL10)
        .filter(ee.Filter.eq('year', col10_year))
        .first()
        .select([band_name])
        .rename('b1')
        .unmask(0)
        .byte()
    )


def build_v2(year, region_id, region_geom):
    """
    Constrói a imagem v2 para (year, region_id) aplicando as regras em ordem de prioridade.
    Retorna ee.Image com banda 'b1', clipped à region_geom.
    """
    zero_img = ee.Image(0).clip(region_geom).rename('b1').byte()

    # Regra 1 — Excluir: toda a região vira zero em todos os anos
    if region_id in EXCLUDE_REGIONS:
        return zero_img

    # Regra 3 — Zero back: anos <= zero_from → zero
    zero_from = ZERO_BACK_REGIONS.get(region_id)
    if zero_from is not None and year <= zero_from:
        return zero_img

    # Carrega v1 (com fallback para anos ausentes)
    fill_year   = MISSING_V1_FILL.get((region_id, year), year)
    asset_id_v1 = f'{ASSET_SOURCE}/solar-panel-{fill_year}-{region_id}-{VERSION_IN}'
    v1 = ee.Image(asset_id_v1).select('b1').unmask(0).byte()

    # Regra 2 — Merge col10: col10.add(col11).gt(0)
    if region_id in MERGE_COL10_REGIONS:
        col10 = get_col10_image(year)
        return col10.add(v1).gt(0).byte().rename('b1').clip(region_geom)

    # Sem regra especial — cópia direta de v1
    return v1.clip(region_geom).rename('b1')


def export_v2(year, region_id, region_geom):
    asset_name = f'solar-panel-{year}-{region_id}-{VERSION_OUT}'
    asset_id   = f'{ASSET_SOURCE}/{asset_name}'

    if SKIP_EXISTING and asset_exists(asset_id):
        print(f'    [{year}] já existe — pulando.')
        return

    img = build_v2(year, region_id, region_geom)

    img_final = img.selfMask().set({
        'year':          year,
        'collection_id': 11,
        'theme':         'SOLAR-PANELS',
        'source':        'geodatin',
        'version':       VERSION_OUT,
        'territory':     'BRAZIL',
        'region':        region_id,
    })

    task = ee.batch.Export.image.toAsset(
        image=img_final,
        description=asset_name,
        assetId=asset_id,
        scale=SCALE,
        crs=CRS,
        region=region_geom,
        maxPixels=1e13,
        pyramidingPolicy={'b1': 'mode'},
    )
    task.start()
    print(f'    [{year}] task → {asset_id}')


# ==============================================================================
# PIPELINE — coleta regiões v1, aplica regras, exporta v2
# ==============================================================================

print("Listando assets v1 na coleção fonte...")
all_assets = list_collection_assets(ASSET_SOURCE)

# Coleta os region_ids únicos presentes em v1
region_ids_seen = set()
for asset in all_assets:
    name = asset.get('name', asset.get('id', ''))
    year, region_id, version = parse_asset_name(name)
    if version == VERSION_IN and year in YEARS and region_id:
        region_ids_seen.add(region_id)

region_ids = sorted(region_ids_seen)
print(f"Regiões encontradas (v{VERSION_IN}): {len(region_ids)}")

shp_regions = ee.FeatureCollection(ASSET_SHP_REGIONS)

print(f"\nAnos         : {YEARS}")
print(f"Versão       : {VERSION_IN} → {VERSION_OUT}")
print(f"Excluir      : {len(EXCLUDE_REGIONS)} regiões")
print(f"Merge col10  : {len(MERGE_COL10_REGIONS)} regiões")
print(f"Zero back    : {len(ZERO_BACK_REGIONS)} regiões")
print("=" * 60)

# sys.exit()  # descomente para inspecionar sem disparar tasks

for cc, region_id in enumerate(region_ids):
    region_geom = (
        shp_regions
        .filter(ee.Filter.eq('system:index', region_id))
        .first()
        .geometry()
    )

    # Resumo da regra para log
    rules = []
    if region_id in EXCLUDE_REGIONS:
        rules.append('EXCLUIR')
    if region_id in MERGE_COL10_REGIONS:
        rules.append('MERGE-COL10')
    if region_id in ZERO_BACK_REGIONS:
        rules.append(f'ZERO-BACK≤{ZERO_BACK_REGIONS[region_id]}')
    if not rules:
        rules.append('cópia v1')

    print(f"\n#{cc + 1:>3}/{len(region_ids)} — {region_id}  [{', '.join(rules)}]")

    for year in YEARS:
        export_v2(year, region_id, region_geom)

print("\n" + "=" * 60)
print("Todas as exportações foram disparadas.")
print("Acompanhe as tasks no Earth Engine Code Editor → Tasks.")
