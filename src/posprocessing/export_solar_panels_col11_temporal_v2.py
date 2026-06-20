#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exporta camadas anuais de FV com consistência temporal — Coleção 11 v2
=======================================================================
Regra central: um pixel só pode existir no ano N se também existe no ano N+1.
  2025 v2 = 2025 v1  (referência, sem filtro)
  2024 v2 = 2024 v1  ∩  2025 v2
  2023 v2 = 2023 v1  ∩  2024 v2
  ...
  2016 v2 = 2016 v1  ∩  2017 v2

Entrada : projects/mapbiomas-workspace/AMOSTRAS/col11/CAATINGA/layers_energia  (versão 1)
Saída   : mesmo ImageCollection, versão 2
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

ASSET_COLLECTION    = 'projects/mapbiomas-workspace/AMOSTRAS/col11/CAATINGA/layers_energia'
ASSET_LIMITE_BRASIL = 'users/CartasSol/shapes/Brasil_Manual'

VERSION_IN  = 1
VERSION_OUT = 2
SCALE       = 4.77        # m/pixel — Planet NICFI
CRS         = 'EPSG:4326'
YEARS_DESC  = list(range(2025, 2015, -1))   # [2025, 2024, ..., 2016]
SKIP_EXISTING = True

limite_brasil = ee.FeatureCollection(ASSET_LIMITE_BRASIL).geometry()

# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================

def asset_exists(asset_id: str) -> bool:
    try:
        ee.data.getAsset(asset_id)
        return True
    except Exception:
        return False


def load_v1(year: int) -> ee.Image:
    asset_id = f'{ASSET_COLLECTION}/solar-panel-{year}-{VERSION_IN}'
    return ee.Image(asset_id).select('b1').selfMask()


def load_v2(year: int) -> ee.Image:
    asset_id = f'{ASSET_COLLECTION}/solar-panel-{year}-{VERSION_OUT}'
    return ee.Image(asset_id).select('b1').selfMask()


def export_image(img: ee.Image, year: int) -> None:
    asset_name = f'solar-panel-{year}-{VERSION_OUT}'
    asset_id   = f'{ASSET_COLLECTION}/{asset_name}'

    img_out = (
        img
        .clip(limite_brasil)
        .selfMask()
        .byte()
        .rename('b1')
        .set({
            'year':          year,
            'collection_id': 11,
            'theme':         'SOLAR-PANELS',
            'source':        'geodatin',
            'version':       VERSION_OUT,
            'territory':     'BRAZIL',
        })
    )

    task = ee.batch.Export.image.toAsset(
        image=img_out,
        description=asset_name,
        assetId=asset_id,
        scale=SCALE,
        crs=CRS,
        region=limite_brasil,
        maxPixels=1e13,
        pyramidingPolicy={'b1': 'mode'},
    )
    task.start()
    print(f'  [v2] task iniciada → {asset_id}')


# ==============================================================================
# PIPELINE — ordem decrescente para construir a cadeia temporal
# ==============================================================================

print(f"Anos (decrescente): {YEARS_DESC}")
print(f"Collection        : {ASSET_COLLECTION}")
print(f"Versão entrada/saída: {VERSION_IN} → {VERSION_OUT}")
print("=" * 60)

prev_filtered = None   # imagem v2 do ano anterior (ano mais recente processado)

for year in YEARS_DESC:
    v2_id = f'{ASSET_COLLECTION}/solar-panel-{year}-{VERSION_OUT}'
    print(f"\n[{year}]")

    if year == 2025:
        # Ano de referência: v2 = v1 sem filtro temporal
        filtered = load_v1(year)
        print(f"  2025: referência — sem filtro temporal (v1 direto)")
    else:
        # Aplica consistência: mantém só pixels que existem no ano seguinte
        v1 = load_v1(year)
        # prev_filtered é a imagem v2 do ano+1 já calculada (ou carregada do GEE)
        filtered = v1.updateMask(prev_filtered.unmask(0))
        print(f"  {year}: v1 ∩ {year + 1} v2")

    if SKIP_EXISTING and asset_exists(v2_id):
        print(f"  asset v2 já existe — pulando exportação.")
        # Carrega o v2 exportado para manter a cadeia consistente
        prev_filtered = load_v2(year)
    else:
        export_image(filtered, year)
        # Usa a expressão computada (não o asset exportado) para continuar a cadeia
        # enquanto as tasks ainda estão rodando em paralelo no GEE
        prev_filtered = filtered

print("\n" + "=" * 60)
print("Todas as exportações foram disparadas.")
print("Acompanhe as tasks no Earth Engine Code Editor → Tasks.")
