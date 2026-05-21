#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pré-exporta Labels FV 2024 como Image Assets no GEE
=====================================================
Para cada região em ASSET_REGIONS_2024 (filtrada por ASSET_POINT_SAMPLES),
computa o label binário FV 2024 (UNet+ResNet50 + polígonos base + exclusão)
e exporta como Image asset em ASSET_OUTPUT_COLLECTION.

Nome de cada asset: {feat_id_safe}_label_2024_unet_resnet50

ANTES DE RODAR (apenas uma vez):
  earthengine create collection projects/mapbiomas-caatinga-cloud04/assets/rotulos_fv_2024

FLUXO:
  1. Rode este script → gera tasks de Export.image.toAsset para cada região.
  2. Aguarde as tasks concluírem no GEE.
  3. No script principal, ative USE_PRECOMP_LABEL_2024=True — ele carrega
     os assets desta coleção em vez de recomputar o label a cada shard.
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

ASSET_REGIONS_2024       = "projects/mapbiomas-arida/energias/shp_area_fotovoltaic_samples_update_16_05_2026"
ASSET_POINT_SAMPLES      = "projects/mapbiomas-arida/energias/pontos_areas_DB_16_05_2026"
ASSET_LABEL_2024         = "projects/geo-data-s/assets/fotovoltaica/usinas_br_gc"
ASSET_LIMIT_ROTULOS_2024 = "projects/mapbiomas-arida/energias/polygons_base_paneis_fotovoltaicos_16_05_2026"
ASSET_EXCLUSION_2024     = "projects/mapbiomas-arida/energias/poligons_exclusion_comision_16_05_2026"

ASSET_OUTPUT_COLLECTION  = "projects/mapbiomas-caatinga-cloud04/assets/rotulos_fv_2024"

YEAR         = 2024
SCALE        = 4.77          # m/pixel — mesma resolução do NICFI
CRS          = "EPSG:4326"
REGION_INIC  = 10
REGION_END   = 100           # será limitado ao total real de regiões
SKIP_EXISTING = True         # pula regiões cujo asset já existe no GEE

# ==============================================================================
# FUNÇÕES
# ==============================================================================

def asset_exists(asset_id):
    """Retorna True se o asset já existe no GEE."""
    try:
        ee.data.getAsset(asset_id)
        return True
    except Exception:
        return False


def build_label_2024(year, geometry):
    """
    Label binário FV 2024 restrito à geometry:
      - mosaic de usinas_br_gc (unet+resnet50) dentro da região
      - soma polígonos base confirmados (base_FV_complementar)
      - remove máscara de exclusão (comissão)
    """
    fc_rotulos  = ee.FeatureCollection(ASSET_LIMIT_ROTULOS_2024).filterBounds(geometry)
    fc_exclusao = ee.FeatureCollection(ASSET_EXCLUSION_2024).filterBounds(geometry)

    base_FV_complementar = ee.Image(0).paint(fc_rotulos,  1).byte().unmask(0)
    mask_negativa        = ee.Image(0).paint(fc_exclusao, 1).byte().unmask(0)

    return (ee.ImageCollection(ASSET_LABEL_2024)
                .filterBounds(geometry)
                .filter(ee.Filter.eq('modelo', 'unet'))
                .filter(ee.Filter.eq('backbone', 'resnet50'))
                .filter(ee.Filter.neq('formato', 'tfr'))
                .filter(ee.Filter.eq('year', year))
                .mosaic().gte(0.5)
                .add(base_FV_complementar)
                .gte(1)
                .updateMask(mask_negativa.eq(0))
                .unmask(0)
                .rename('label')
                .toByte())


def export_label_asset(label_img, geometry, asset_name, region_id):
    """Submete task Export.image.toAsset para um label de região."""
    asset_id    = f"{ASSET_OUTPUT_COLLECTION}/{asset_name}"
    description = f"label_2024_{region_id}"

    task = ee.batch.Export.image.toAsset(
        image=label_img,
        description=description,
        assetId=asset_id,
        scale=SCALE,
        crs=CRS,
        region=geometry,
        maxPixels=1e10,
        pyramidingPolicy={'label': 'mode'},
    )
    task.start()
    return asset_id


# ==============================================================================
# PIPELINE — itera regiões e exporta label
# ==============================================================================

points_samples = ee.FeatureCollection(ASSET_POINT_SAMPLES)
regions_fc     = ee.FeatureCollection(ASSET_REGIONS_2024).filterBounds(points_samples)
region_list    = regions_fc.toList(regions_fc.size())
total_regions  = regions_fc.size().getInfo()
region_end     = min(total_regions - 1, REGION_END)

print(f"Total de regiões  : {total_regions}")
print(f"Intervalo         : [{REGION_INIC}, {region_end}]")
print(f"Output collection : {ASSET_OUTPUT_COLLECTION}\n")
print("=" * 60)

submetidas  = 0
puladas     = 0
erros       = 0

for global_idx in range(REGION_INIC, region_end + 1):

    feature      = ee.Feature(region_list.get(global_idx))
    geom         = feature.geometry()
    feat_id      = feature.get('system:index').getInfo() or f'{global_idx:04d}'
    feat_id_safe = str(feat_id).replace('/', '_').replace(':', '_')
    asset_name   = f"{feat_id_safe}_label_2024_unet_resnet50"
    asset_id     = f"{ASSET_OUTPUT_COLLECTION}/{asset_name}"

    print(f"[{global_idx + 1:3d}/{total_regions}] {feat_id_safe}", end="  ")

    if SKIP_EXISTING and asset_exists(asset_id):
        print("→ asset já existe. Pulando.")
        puladas += 1
        continue

    try:
        label = build_label_2024(YEAR, geom)
        export_label_asset(label, geom, asset_name, feat_id_safe)
        print(f"→ task submetida: {asset_id}")
        submetidas += 1
    except Exception as e:
        print(f"→ ERRO: {e}")
        erros += 1

print("\n" + "=" * 60)
print(f"Submetidas : {submetidas}")
print(f"Puladas    : {puladas}  (já existiam)")
print(f"Erros      : {erros}")
print("\nAcompanhe as tasks no Earth Engine Code Editor.")
print("Após conclusão, ative USE_PRECOMP_LABEL_2024=True no script principal.")
