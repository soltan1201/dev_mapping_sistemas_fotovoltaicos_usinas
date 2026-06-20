#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exporta camadas anuais de painéis fotovoltaicos — Coleção 11
=============================================================
Para cada ano em YEARS, compõe a imagem final de FV aplicando:
  - modelo: UNet + EfficientNetB7 (ImageCollection predições)
  - colagem de buracos da segmentação (apenas 2025)
  - merge com Col10 por regiões específicas (anos 2024 ee 2025 : merge_col11_col10)
  - substituição por Col10 em regiões com falha do modelo (2024: repetir24)
  - mistura predFV + Col10 para todos os anos 2016–2023
  - máscara de erros de segmentação (exclusion v1 + v2)
  - máscara de regiões sem FV confirmado (excluir_analises)
  - recorte ao limite por region
  - os pixels presente no ano X filtram os pixels presentes do ano X - 1
  - na lista SPECIAL_REGIONS_COL10_2025, nessas regiões va o dados de col10 
      reprojectado e o ano de 2024 se repete para 2025

Saída: ImageCollection   projects/mapbiomas-workspace/AMOSTRAS/col11/CAATINGA/layers_energia
Nome de cada asset    :  solar-panel-{year}-{region_id}-{version}
Regiões especiais     :  11 regiões (5a–64) geram imagem extra em 2025 usando Col10
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

ASSET_PREDICOES      = 'projects/geo-data-s/assets/fotovoltaica/usinas_br_gc'
ASSET_COL10          = 'projects/mapbiomas-brazil/assets/LAND-COVER/COLLECTION-10/SOLAR-PANELS/classification'
ASSET_SHP_REGIONS    = 'projects/mapbiomas-arida/energias/shp_revisao3_10_06_2026_buffer_fotovoltaic_5km'
ASSET_LIMITE_BRASIL  = 'users/CartasSol/shapes/Brasil_Manual'

# camadas de correção
ASSET_CLASSES_2025FV = 'projects/mapbiomas-arida/energias/polygons_base_paneis3_FV_10_06_2026'
ASSET_MERGE_REGIONS  = 'projects/mapbiomas-arida/energias/region_with_merge_layer_col11_col10'
ASSET_EXCLUIR        = 'projects/mapbiomas-arida/energias/regions_to_excluir_analises_10_06_2026'
ASSET_REPETIR24      = 'projects/mapbiomas-arida/energias/region_with_layer_colection10_24_10_06_2026'
ASSET_EXCLUSION_V1   = 'projects/mapbiomas-arida/energias/poligons_exclusion_comision_v1_10_06_2026'
ASSET_EXCLUSION_V2   = 'projects/mapbiomas-arida/energias/poligons_exclusion_comision_v2_10_06_2026'

ASSET_OUTPUT   = 'projects/mapbiomas-workspace/AMOSTRAS/col11/CAATINGA/layers_energia'

MODELO         = 'unet'
BACKBONE       = 'efficientnetb7'
VERSION        = 1
COL10_MAX_YEAR = 2024        # último ano disponível na Coleção 10
COL10_SCALE    = 30          # resolução nativa da Col10 (Landsat)
SCALE          = 4.77        # m/pixel — resolução Planet NICFI (col11 reference)
CRS            = 'EPSG:4326'
YEARS          = list(range(2025, 2015, -1))  # 2025 → 2016 (ordem para filtro temporal)
SKIP_EXISTING  = True        # pula anos cujo asset já existe no GEE

# Regiões que geram imagem extra em 2025 usando apenas Col10
SPECIAL_REGIONS_COL10_2025 = {
    "0000000000000000005a", "0000000000000000005b", "0000000000000000005c",
    "0000000000000000005d", "0000000000000000005e", "0000000000000000005f",
    "00000000000000000060", "00000000000000000061", "00000000000000000062",
    "00000000000000000063", "00000000000000000064",
}

# ==============================================================================
# CAMADAS COMPARTILHADAS (construídas uma vez, reutilizadas em todos os anos)
# ==============================================================================

shp_regions   = ee.FeatureCollection(ASSET_SHP_REGIONS)
limite_brasil = ee.FeatureCollection(ASSET_LIMITE_BRASIL).geometry()

# Máscara de erros de segmentação (v1 + v2 unidos)
img_exclusion = (
    ee.Image(0)
    .paint(ee.FeatureCollection(ASSET_EXCLUSION_V1)
             .merge(ee.FeatureCollection(ASSET_EXCLUSION_V2)), 1)
    .unmask(0)
    .byte()
)   # 1 = pixel errado a remover

# Regiões confirmadas sem FV — apaga tudo dentro delas
excluir_regions = shp_regions.filterBounds(ee.FeatureCollection(ASSET_EXCLUIR))
img_sem_fv = ee.Image(0).paint(excluir_regions, 1).unmask(0).byte()   # 1 = sem FV

# Regiões onde 2024 deve usar Col10 no lugar do predFV
repetir24_regions = shp_regions.filterBounds(ee.FeatureCollection(ASSET_REPETIR24))
img_repetir24 = ee.Image(0).paint(repetir24_regions, 1).unmask(0).byte()   # 1 = usar Col10

# Regiões onde 2025 deve juntar predFV com Col10
merge_regions = shp_regions.filterBounds(ee.FeatureCollection(ASSET_MERGE_REGIONS))
img_merge = ee.Image(0).paint(merge_regions, 1).unmask(0).byte()   # 1 = mesclar Col10

# Polígonos que preenchem buracos da segmentação — apenas 2025
img_classes_2025fv = (
    ee.Image(0)
    .paint(ee.FeatureCollection(ASSET_CLASSES_2025FV), 1)
    .selfMask()
    .rename('b1')
    .byte()
)

# ==============================================================================
# FUNÇÕES AUXILIARES
# ==============================================================================

def asset_exists(asset_id: str) -> bool:
    try:
        ee.data.getAsset(asset_id)
        return True
    except Exception:
        return False


def get_pred_fv(year: int) -> ee.Image:
    return (
        ee.ImageCollection(ASSET_PREDICOES)
        .filter(ee.Filter.eq('year', year))
        .filter(ee.Filter.eq('modelo', MODELO))
        .filter(ee.Filter.eq('backbone', BACKBONE))
        .mosaic()
        .toFloat()       # normaliza tipo antes de entrar em mosaicos mistos
        .selfMask()
        .rename('b1')
    )


def get_pred_col10(year: int) -> ee.Image:
    col10_year = min(year, COL10_MAX_YEAR)
    # Col10 nativa a 30 m (Integer/Landsat); reprojetamos para SCALE (4.77 m)
    # e convertemos para Float para compatibilidade de tipo com predFV.
    return (
        ee.ImageCollection(ASSET_COL10)
        .filter(ee.Filter.eq('year', col10_year))
        .first()
        .toFloat()       # Integer<0,65535> → Float, homogêneo com predFV
        .selfMask()
        .rename('b1')
        .reproject(crs=CRS, scale=SCALE)
    )


def build_image(year: int, region_geom: ee.Geometry) -> ee.Image:
    fv  = get_pred_fv(year)
    c10 = get_pred_col10(year)

    if year == 2025:
        # Preenche buracos da segmentação com polígonos desenhados
        fv = ee.ImageCollection([fv, img_classes_2025fv]).mosaic()
        # Nas regiões de merge: col10 preenche onde predFV não detectou
        c10_merge = c10.updateMask(img_merge)
        img = ee.ImageCollection([fv, c10_merge]).mosaic()

    elif year == 2024:
        # Nas regiões repetir24: col10 substitui predFV (modelo ficou pior)
        # Nas regiões merge: col10 preenche onde predFV não detectou (igual 2025)
        fv_fora    = fv.updateMask(img_repetir24.Not())
        c10_dentro = c10.updateMask(img_repetir24)
        c10_merge  = c10.updateMask(img_merge)
        img = ee.ImageCollection([fv_fora, c10_dentro, c10_merge]).mosaic()

    else:  # 2016–2023: predFV + Col10 misturados (predFV tem prioridade)
        img = ee.ImageCollection([fv, c10]).mosaic()

    # Remove pixels erroneamente segmentados (exclusion v1 + v2)
    img = img.updateMask(img_exclusion.Not())

    # Remove todos os pixels em regiões confirmadas sem FV
    img = img.updateMask(img_sem_fv.Not())

    return img.clip(region_geom).selfMask().byte().rename('b1')


def export_year_region(
    year: int,
    region_id: str,
    region_geom: ee.Geometry,
    prev_filtered: ee.Image,
) -> ee.Image:
    """
    Exporta um ano para uma região e retorna a imagem para encadear no próximo ano.
    prev_filtered: imagem filtrada do ano+1 (None para 2025).
    """
    asset_name = f'solar-panel-{year}-{region_id}-{VERSION}'
    asset_id   = f'{ASSET_OUTPUT}/{asset_name}'

    img = build_image(year, region_geom)

    # Consistência temporal: remove pixels ausentes no ano seguinte
    if prev_filtered is not None:
        img = img.updateMask(prev_filtered.unmask(0))

    if SKIP_EXISTING and asset_exists(asset_id):
        print(f'    [{year}] já existe — pulando.')
        return ee.Image(asset_id).select('b1').selfMask()

    img_final = img.set({
        'year':          year,
        'collection_id': 2,
        'theme':         'SOLAR-PANELS',
        'source':        'geodatin',
        'version':       VERSION,
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
    return img


def export_col10_extra(region_id: str, region_geom: ee.Geometry) -> None:
    """Imagem extra para 2025 usando apenas Col10 (regiões especiais 5a–64)."""
    asset_name = f'solar-panel-2025-{region_id}-{VERSION}'
    asset_id   = f'{ASSET_OUTPUT}/{asset_name}'

    if SKIP_EXISTING and asset_exists(asset_id):
        print('    [2025-col10] já existe — pulando.')
        return

    img = (
        get_pred_col10(2025)
        .clip(region_geom)
        .selfMask()
        .byte()
        .rename('b1')
        .set({
            'year':          2025,
            'collection_id': 11,
            'theme':         'SOLAR-PANELS',
            'source':        'geodatin',
            'version':       VERSION,
            'territory':     'BRAZIL',
            'region':        region_id,
        })
    )

    task = ee.batch.Export.image.toAsset(
        image=img,
        description=asset_name,
        assetId=asset_id,
        scale=SCALE,
        crs=CRS,
        region=region_geom,
        maxPixels=1e13,
        pyramidingPolicy={'b1': 'mode'},
    )
    task.start()
    print(f'    [2025-col10] task → {asset_id}')


# ==============================================================================
# PIPELINE — carrega regiões, exclui inválidas, exporta por região × ano
# ==============================================================================

print("Carregando regiões válidas...")
valid_regions_fc = ee.FeatureCollection(ASSET_SHP_REGIONS).filter(
    ee.Filter.bounds(ee.FeatureCollection(ASSET_EXCLUIR)).Not()
)
region_features = valid_regions_fc.getInfo()['features']
numero_regions = len(region_features)
print(f"Regiões válidas   : {numero_regions}")
print(f"Anos (decrescente): {YEARS}")
print(f"Output collection : {ASSET_OUTPUT}")
print(f"Modelo/backbone   : {MODELO} / {BACKBONE}")
print(f"Versão            : {VERSION}")
print("=" * 60)
# sys.exit()
pos_inic = 2
pos_end =  100
for cc, feat in enumerate(region_features[pos_inic:pos_end]):
    region_id   = feat['id']   # system:index da FeatureCollection
    region_geom = ee.Geometry(feat['geometry'])
    print(f"\n #{cc + pos_inic}/{numero_regions} >>> Região {region_id}")

    prev_filtered = None   # 2025 é a referência — sem filtro temporal
    for year in YEARS:
        prev_filtered = export_year_region(year, region_id, region_geom, prev_filtered)

    if region_id in SPECIAL_REGIONS_COL10_2025:
        print("  → região especial: exportando col10 extra para 2025")
        export_col10_extra(region_id, region_geom)

print("\n" + "=" * 60)
print("Todas as exportações foram disparadas.")
print("Acompanhe as tasks no Earth Engine Code Editor → Tasks.")

print("=" * 60)
print("Todas as exportações foram disparadas.")
print("Acompanhe as tasks no Earth Engine Code Editor → Tasks.")
