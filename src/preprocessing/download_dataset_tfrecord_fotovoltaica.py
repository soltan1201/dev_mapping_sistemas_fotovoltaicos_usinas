#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download Dataset TFRecord - Fotovoltaica (Planet NICFI)
========================================================
Exporta patches 257×257 (kernel rect 128) de 8 bandas + label como TFRecord no Drive.

COMO USAR — duas rodadas separadas:
  Rodada 1:  YEARS = [2024]       → regiões shp_buffer_fotovoltaic_5km_samples_2024
  Rodada 2:  YEARS = [2022, 2023] → regiões shp_buffer_fotovoltaic_5km_samples_22_24
  A seleção do asset de regiões é automática com base em YEARS.

PIPELINE A — Grade regular nas regiões buffer 5 km:
  Para cada região × ano: gera grade com STRIDE_PIXELS, exporta shards TFRecord.

PIPELINE B — Patches extras de balanceamento (FV focus):
  Por ano: amostra N_EXTRA_PATCHES aleatórios dentro de shp_area_fotovoltaic_samples.
  Garante referências positivas suficientes frente ao excesso de patches sem FV
  gerado pelos buffers de 5 km.

LABEL:
  Pixels da ImageCollection ASSET_LABEL são restritos a polígonos base
  (shp_polygons_base_paneis_fotovoltaicos_col11). Fora deles → forçado 0,
  eliminando classificações espúrias fora das manchas FV confirmadas.
"""

import sys
import os
import math
import ee
from pathlib import Path

pathparent = str(Path(os.getcwd()).parents[0])
sys.path.append(pathparent)
from configure_account_projects_ee import get_current_account

projAccount = get_current_account()
print(f"Projeto selecionado >>> {projAccount} <<<")

try:
    ee.Initialize(project=projAccount)
    print('Earth Engine inicializado com sucesso!')
except Exception as e:
    print("Erro de Inicialização:", e)
    raise

# ==============================================================================
# 1. CONFIGURAÇÕES
# ==============================================================================

# ── Anos de coleta ─────────────────────────────────────────────────────────────
# Rodada 1: YEARS = [2024]
# Rodada 2: YEARS = [2022, 2023]
YEARS = [2022, 2023]   # [2024] # 

# ── Assets de regiões buffer 5 km (seleção automática pelo YEARS) ──────────────
ASSET_REGIONS_2024  = "projects/mapbiomas-arida/shp_buffer_fotovoltaic_5km_samples_2024"
ASSET_REGIONS_22_24 = "projects/mapbiomas-arida/shp_buffer_fotovoltaic_5km_samples_22_24"

# ── Máscara de rótulos: limita pixels FV a polígonos base confirmados ──────────
ASSET_LIMIT_ROTULOS = "projects/mapbiomas-arida/shp_polygons_base_paneis_fotovoltaicos_col11"

# ── Áreas focus para patches extras de balanceamento ──────────────────────────
ASSET_FV_FOCUS = "projects/mapbiomas-arida/shp_area_fotovoltaic_samples"

# ── ImageCollection de rótulos e mosaico NICFI ────────────────────────────────
ASSET_LABEL = 'projects/geo-data-s/assets/fotovoltaica/version_2_clean'
ASSET_NICFI = 'projects/planet-nicfi/assets/basemaps/americas'

# ── Saída no Google Drive ─────────────────────────────────────────────────────
EXPORT_DRIVE_FOLDER = "DS_FV_NICFI_TFRECORDS"

# ── Parâmetros de patch e grade ───────────────────────────────────────────────
PATCH_SIZE       = 256   # pixels; kernel rect(128,128) → saída 257×257
SCALE            = 4.77  # m/pixel (resolução nativa NICFI Planet)
STRIDE_PIXELS    = 230   # espaçamento da grade em pixels (~1097 m entre pontos)
MAX_PATCHES_FILE = 50    # máx patches por shard (>100 pode estourar o GEE: ~240 MB)

# ── Extra patches balanceamento ───────────────────────────────────────────────
N_EXTRA_PATCHES = 75     # pontos aleatórios por ano no ASSET_FV_FOCUS (20–30)

# ── Intervalo de regiões para processar em lotes (inclusive em ambas as pontas) ─
REGION_INIC = 20 
REGION_END  = 40

# ── Normalização espectral NICFI ──────────────────────────────────────────────
dict_percentil = {
    "blue":  [100,   800],
    "green": [300,  1200],
    "red":   [176,  1700],
    "nir":   [350,  4000],
}

NICFI_BANDS_SRC = ['B', 'G', 'R', 'N']
NICFI_BANDS_DST = ['blue', 'green', 'red', 'nir']
ALL_BANDS       = ['blue', 'green', 'red', 'nir', 'pvi', 'iia', 'ri', 'evi']
SELECTORS       = ALL_BANDS + ['label']

# ── Seleção automática do asset de regiões ────────────────────────────────────
ASSET_REGIONS = ASSET_REGIONS_2024 if YEARS == [2024] else ASSET_REGIONS_22_24
print(f"\nAnos configurados : {YEARS}")
print(f"Asset de regiões  : {ASSET_REGIONS}")


# ==============================================================================
# 2. FUNÇÕES
# ==============================================================================

def build_nicfi_mosaic(year, geometry):
    """
    Mosaico mediana jul–dez do NICFI Planet com 8 bandas em [0, 10000] (Int16).

    Bandas: blue, green, red, nir → normalizadas por percentil
            pvi, iia, ri, evi     → índices espectrais normalizados
    """
    start = f'{year}-07-01'
    end   = f'{year + 1}-01-01'

    mosaic = (ee.ImageCollection(ASSET_NICFI)
              .filterDate(start, end)
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

    # PVI ∈ [-1,1] → [0, 10000]
    pvi = (mosaic.expression(
               'float(BLUE - NIR) / float(BLUE + NIR + 1)',
               {'BLUE': mosaic.select('blue'), 'NIR': mosaic.select('nir')})
           .add(1).divide(2).multiply(10000).toInt16().rename('pvi'))

    # IIA ∈ [-1,1] → [0, 10000]
    iia = (mosaic.expression(
               "float((green - 4 * nir) / (green + 4 * nir + 1))",
               {'green': mosaic.select('green'), 'nir': mosaic.select('nir')})
           .add(1).divide(2).multiply(10000).toInt16().rename('iia'))

    # RI ∈ [-2.4,2.4] → [0, 10000]
    ri = (mosaic.expression(
              "float(2.4 * (red - green) / (red + green + 1))",
              {'red': mosaic.select('red'), 'green': mosaic.select('green')})
          .add(2.4).divide(4.8).multiply(10000).toInt16().rename('ri'))

    # EVI2 ∈ [-2.4,2.4] → [0, 10000]
    evi = (mosaic.expression(
               "float(2.4 * (nir - red) / (1 + nir + red))",
               {'nir': mosaic.select('nir'), 'red': mosaic.select('red')})
           .add(2.4).divide(4.8).multiply(10000).toInt16().rename('evi'))

    return (scaled_img.multiply(10000).toInt16()
            .addBands(pvi).addBands(iia).addBands(ri).addBands(evi))


def build_label(year):
    """
    Rótulo binário FV para o ano, mascarado pelos polígonos base.

    Pixels classificados como FV (>=1) na ImageCollection são aceitos apenas
    se estiverem dentro de shp_polygons_base_paneis_fotovoltaicos_col11.
    Fora dos polígonos base → forçado 0 (elimina classificações espúrias).
    """
    base_mask = (ee.Image(0)
                 .paint(ee.FeatureCollection(ASSET_LIMIT_ROTULOS), 1)
                 .byte())

    return (ee.ImageCollection(ASSET_LABEL)
              .filter(ee.Filter.eq('year', year))
              .mosaic()
              .gte(1)
              .updateMask(base_mask)
              .unmask(0)
              .rename('label')
              .toByte())


def generate_grid_points(geometry, scale_m, stride_pixels):
    """Grade regular de pontos cobrindo a geometria em EPSG:3857."""
    proj         = ee.Projection('EPSG:3857').atScale(scale_m)
    pixel_coords = ee.Image.pixelCoordinates(proj)
    x_idx        = pixel_coords.select('x').divide(scale_m).round().toInt()
    y_idx        = pixel_coords.select('y').divide(scale_m).round().toInt()
    grid_mask    = x_idx.mod(stride_pixels).eq(0).And(y_idx.mod(stride_pixels).eq(0))

    return (grid_mask.selfMask()
            .sample(region=geometry, scale=scale_m, projection=proj,
                    geometries=True, tileScale=4))


def export_shard(patches_array, points, year, tag, shard_idx):
    """
    Extrai patches de 'patches_array' nos 'points' e exporta TFRecord para o Drive.
    Retorna o nome da task criada.
    """
    fname = f"tfrecord_fv_{tag}_{year}_part{shard_idx:03d}"
    task = ee.batch.Export.table.toDrive(
        collection=patches_array.sampleRegions(
            collection=points,
            scale=SCALE,
            geometries=False,
            tileScale=16,
        ),
        description=fname,
        folder=EXPORT_DRIVE_FOLDER,
        fileFormat='TFRecord',
        selectors=SELECTORS,
    )
    task.start()
    return fname


def build_patches_array(year, geometry):
    """Stack completo (8 bandas + label) com neighborhoodToArray pronto para exportar."""
    mosaic = build_nicfi_mosaic(year, geometry)
    label  = build_label(year)
    kernel = ee.Kernel.rectangle(PATCH_SIZE // 2, PATCH_SIZE // 2, 'pixels')

    full_stack = (mosaic.select(ALL_BANDS)
                  .unmask(0)
                  .addBands(label)
                  .toInt16())

    return full_stack.neighborhoodToArray(kernel)


# ==============================================================================
# 3. PIPELINE A — Grade de pontos nas regiões buffer 5 km
# ==============================================================================

print("\n" + "=" * 60)
print("PIPELINE A — Regiões buffer 5 km (grade regular)")
print("=" * 60)

regions_fc    = ee.FeatureCollection(ASSET_REGIONS)
region_list   = regions_fc.toList(regions_fc.size())
total_regions = regions_fc.size().getInfo()
region_end    = min(total_regions - 1, REGION_END)
fv_focus_fc   = ee.FeatureCollection(ASSET_FV_FOCUS)

print(f"Total de regiões no asset : {total_regions}")
print(f"Intervalo processado      : [{REGION_INIC}, {region_end}] "
      f"({region_end - REGION_INIC + 1} regiões)")

for global_idx in range(REGION_INIC, region_end + 1):

    feature      = ee.Feature(region_list.get(global_idx))
    geom         = feature.geometry()
    feat_id      = feature.get('system:index').getInfo() or f'{global_idx:04d}'
    feat_id_safe = str(feat_id).replace('/', '_').replace(':', '_')

    print(f"\n{'=' * 60}")
    print(f"[{global_idx + 1}/{total_regions}] Região: {feat_id_safe}")

    grid_points         = generate_grid_points(geom, SCALE, STRIDE_PIXELS)
    n_points            = grid_points.size().getInfo()
    fv_focus_local      = fv_focus_fc.filterBounds(geom)
    fv_focus_local_geom = fv_focus_local.geometry()
    print(f"  Grade: {n_points} pontos  "
          f"(stride={STRIDE_PIXELS}px ≈ {STRIDE_PIXELS * SCALE:.0f} m)")

    if n_points == 0:
        print("  Região sem pontos na grade. Pulando.")
        continue

    for year in YEARS:
        print(f"\n  --- Ano {year} ---")

        extra_points = (ee.Image.constant(1)
                        .sample(
                            region=fv_focus_local_geom,
                            numPixels=N_EXTRA_PATCHES,
                            scale=SCALE,
                            geometries=True,
                            seed=global_idx * 1000 + year * 137,
                            tileScale=4,
                        ))
        n_extra    = extra_points.size().getInfo()
        all_points = grid_points.merge(extra_points)
        total_pts  = n_points + n_extra
        print(f"  Grade: {n_points} pts  FV focus: {n_extra} pts  Total: {total_pts}")

        patches_array = build_patches_array(year, geom)
        num_shards    = max(1, math.ceil(total_pts / MAX_PATCHES_FILE))
        sample_points = all_points.randomColumn('shard_idx', seed=year)

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
                fname = export_shard(patches_array, shard_pts, year, feat_id_safe, i)
                print(f"    Shard {i:03d}: {n_shard} pts → {fname}")
            except Exception as e:
                print(f"    Erro shard {i:03d}: {e}")


# ==============================================================================
# 4. PIPELINE B — Patches extras de balanceamento (FV focus areas)
# ==============================================================================

print("\n" + "=" * 60)
print("PIPELINE B — Extra patches FV (balanceamento)")
print(f"  Asset: {ASSET_FV_FOCUS}")
print(f"  Patches extras por ano: {N_EXTRA_PATCHES}")
print("=" * 60)

# fv_focus_geom = fv_focus_fc.geometry()

# for year in YEARS:
#     print(f"\n  --- Ano {year} ---")

#     extra_points = (ee.Image.constant(1)
#                     .sample(
#                         region=fv_focus_geom,
#                         numPixels=N_EXTRA_PATCHES,
#                         scale=SCALE,
#                         geometries=True,
#                         seed=year * 137,
#                         tileScale=4,
#                     ))

#     n_extra = extra_points.size().getInfo()
#     print(f"  Pontos amostrados: {n_extra}")

#     if n_extra == 0:
#         print("  Nenhum ponto retornado. Verifique ASSET_FV_FOCUS.")
#         continue

#     try:
#         patches_array = build_patches_array(year, fv_focus_geom)
#         fname = export_shard(patches_array, extra_points, year, 'focus_extra', 0)
#         print(f"  Task enviada: {fname}")
#     except Exception as e:
#         print(f"  Erro no export extra ({year}): {e}")

print("\nTodos os pipelines concluídos. Verifique as Tasks no Earth Engine Code Editor.")
