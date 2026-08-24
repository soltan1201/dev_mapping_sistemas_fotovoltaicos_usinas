#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download Dataset TFRecord - Fotovoltaica (Planet NICFI)
=======================================================
Exporta patches TFRecord com mosaico NICFI + rótulo FV para treino.

Rótulos: usa ASSET_LABEL para todos os anos (2022, 2023, 2024, 2025).
  → projects/mapbiomas-workspace/AMOSTRAS/col11/CAATINGA/layers_energia

Regiões:
  2022/2023 → ASSET_REGIONS_22_24
  2024/2025 → ASSET_REGIONS_2024

COMO USAR:
  Ajuste YEARS e REGION_INIC/REGION_END conforme necessário, depois rode.
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
YEARS = [2022, 2023, 2024, 2025]  # [2022, 2023]

# ── Assets de regiões (seleção automática pelo YEARS) ─────────────────────────
ASSET_REGIONS_2024  = "projects/mapbiomas-arida/energias/shp_area_fotovoltaic_samples_update_16_05_2026"
ASSET_REGIONS_22_24 = "projects/mapbiomas-arida/shp_buffer_fotovoltaic_5km_samples_22_24"

# ── Máscara de rótulos (usada apenas para 2022/2023) ──────────────────────────
ASSET_LIMIT_ROTULOS      = "projects/mapbiomas-arida/shp_polygons_base_paneis_fotovoltaicos_col11"
ASSET_LIMIT_ROTULOS_2024 = "projects/mapbiomas-arida/energias/polygons_base_paneis_fotovoltaicos_16_05_2026"

# ── Áreas focus para patches extras de balanceamento ──────────────────────────
ASSET_FV_FOCUS      = "projects/mapbiomas-arida/shp_area_fotovoltaic_samples"
ASSET_POINT_SAMPLES = "projects/mapbiomas-arida/energias/pontos_areas_DB_16_05_2026"

# ── ImageCollection de rótulos e mosaico NICFI ────────────────────────────────
ASSET_LABEL      = 'projects/mapbiomas-workspace/AMOSTRAS/col11/CAATINGA/layers_energia'
# ASSET_LABEL    = 'projects/geo-data-s/assets/fotovoltaica/version_2_clean'
ASSET_NICFI      = 'projects/planet-nicfi/assets/basemaps/americas'

# ── Saída no Google Drive ─────────────────────────────────────────────────────
EXPORT_DRIVE_FOLDER = "DS_FV_NICFI_TFR_V3"

# ── Parâmetros de patch e grade ───────────────────────────────────────────────
PATCH_SIZE       = 256
SCALE            = 4.77   # m/pixel (resolução nativa NICFI Planet)
STRIDE_PIXELS    = 230    # ~1097 m entre pontos da grade
MAX_PATCHES_FILE = 50     # máx patches por shard

# ── Extra patches balanceamento ───────────────────────────────────────────────
N_EXTRA_PATCHES = 55

# ── Intervalo de regiões (inclusive em ambas as pontas) ───────────────────────
REGION_INIC = 70
REGION_END  = 80

# ── Pula tasks já COMPLETED no GEE ───────────────────────────────────────────
SKIP_COMPLETED = True

# ── Normalização espectral NICFI ──────────────────────────────────────────────
dict_percentil = {
    "blue":  [100,   800],
    "green": [300,  1200],
    "red":   [176,  1700],
    "nir":   [350,  4000],
}

NICFI_BANDS_SRC = ['B', 'G', 'R', 'N']
NICFI_BANDS_DST = ['blue', 'green', 'red', 'nir']
ALL_BANDS       = ['blue', 'green', 'red', 'pvi', 'pvpi']
SELECTORS       = ALL_BANDS + ['label']

# ── Seleção automática do asset de regiões ────────────────────────────────────
# 2022/2023 → grade antiga; 2024/2025 → grade atualizada
ASSET_REGIONS = ASSET_REGIONS_22_24 if set(YEARS).issubset({2022, 2023}) else ASSET_REGIONS_2024
print(f"\nAnos configurados : {YEARS}")
print(f"Asset de regiões  : {ASSET_REGIONS}")


# ==============================================================================
# CARREGA TASKS JÁ CONCLUÍDAS (quando SKIP_COMPLETED=True)
# ==============================================================================

def get_completed_tasks(prefix="tfrecord_fv_"):
    """Retorna um set com os descriptions de todas as tasks COMPLETED no GEE."""
    try:
        all_tasks = ee.data.getTaskList()
        completed = {
            t["description"]
            for t in all_tasks
            if t.get("state") == "COMPLETED"
            and t.get("description", "").startswith(prefix)
        }
        print(f"  {len(completed)} tasks já COMPLETED encontradas no GEE.")
        return completed
    except Exception as e:
        print(f"  Aviso: não foi possível buscar tasks concluídas: {e}")
        return set()


if SKIP_COMPLETED:
    print("\nSKIP_COMPLETED=True → buscando tasks já concluídas no GEE...")
    _completed_tasks = get_completed_tasks(prefix="tfrecord_fv_")
else:
    _completed_tasks = set()


# ==============================================================================
# 2. FUNÇÕES
# ==============================================================================

def build_nicfi_mosaic(year, geometry):
    """Mosaico mediana jul–dez do NICFI Planet com 5 bandas normalizadas."""
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

    pvi = (mosaic.expression(
               'float(BLUE - NIR) / float(BLUE + NIR + 1)',
               {'BLUE': mosaic.select('blue'), 'NIR': mosaic.select('nir')})
            .add(1).divide(2).multiply(10000).toInt16().rename('pvi'))

    pvpi = (mosaic.expression(
                'float((green - blue) / (green + blue))',
                {'green': mosaic.select('green'), 'blue': mosaic.select('blue')})
                .add(1).divide(2).multiply(10000).toInt16().rename('pvpi'))

    return (scaled_img.multiply(10000).toInt16().select(['blue', 'green', 'red'])
            .addBands(pvi).addBands(pvpi))


def build_label(year):
    """Rótulo binário FV para qualquer ano (2022–2025) via ASSET_LABEL."""
    base_mask = (ee.Image(0)
                 .paint(ee.FeatureCollection(ASSET_LIMIT_ROTULOS_2024), 1)
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
    proj      = ee.Projection('EPSG:3857').atScale(scale_m)
    px_coords = ee.Image.pixelCoordinates(proj)
    x_idx     = px_coords.select('x').divide(scale_m).round().toInt()
    y_idx     = px_coords.select('y').divide(scale_m).round().toInt()
    grid_mask = x_idx.mod(stride_pixels).eq(0).And(y_idx.mod(stride_pixels).eq(0))
    return (grid_mask.selfMask()
            .sample(region=geometry, scale=scale_m, projection=proj,
                    geometries=True, tileScale=4))


def export_shard(patches_array, points, year, tag, shard_idx):
    """Extrai patches nos points e exporta TFRecord para o Drive."""
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


def build_patches_array(year, geometry, feat_id_safe):
    """Stack completo (5 bandas + label) com neighborhoodToArray."""
    mosaic = build_nicfi_mosaic(year, geometry)
    label  = build_label(year)

    kernel = ee.Kernel.rectangle(PATCH_SIZE // 2, PATCH_SIZE // 2, 'pixels')

    # Buffer de meio patch para garantir vizinhança completa nos patches de borda.
    # clip() antes de unmask() limita o footprint ao buffer e evita que unmask()
    # expanda o cálculo para os tiles completos do NICFI (causa do export de 4 h).
    roi = geometry.buffer(PATCH_SIZE * SCALE)

    full_stack = (mosaic.select(ALL_BANDS)
                  .clip(roi)
                  .unmask(0)
                  .addBands(label.clip(roi).unmask(0))
                  .toInt16())

    return full_stack.neighborhoodToArray(kernel)


# ==============================================================================
# 3. PIPELINE A — Grade de pontos nas regiões
# ==============================================================================

print("\n" + "=" * 60)
print("PIPELINE A — Regiões (grade regular)")
print("=" * 60)

points_samples = ee.FeatureCollection(ASSET_POINT_SAMPLES)
regions_fc     = ee.FeatureCollection(ASSET_REGIONS).filterBounds(points_samples)
region_list    = regions_fc.toList(regions_fc.size())
total_regions  = regions_fc.size().getInfo()
region_end     = min(total_regions - 1, REGION_END)
fv_focus_fc    = ee.FeatureCollection(ASSET_FV_FOCUS)

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
        all_points = grid_points.merge(extra_points)
        total_pts  = n_points + N_EXTRA_PATCHES  # estimativa; .sample pode retornar menos
        print(f"  Grade: {n_points} pts  FV focus: ~{N_EXTRA_PATCHES} pts  Total: ~{total_pts}")

        patches_array = build_patches_array(year, geom, feat_id_safe)
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

            fname_preview = f"tfrecord_fv_{feat_id_safe}_{year}_part{i:03d}"
            if SKIP_COMPLETED and fname_preview in _completed_tasks:
                print(f"    Shard {i:03d}: já COMPLETED → {fname_preview}. Pulando.")
                continue

            try:
                fname = export_shard(patches_array, shard_pts, year, feat_id_safe, i)
                print(f"    Shard {i:03d}: enviado → {fname}")
            except Exception as e:
                print(f"    Erro shard {i:03d}: {e}")


# ==============================================================================
# 4. PIPELINE B — Patches extras de balanceamento (FV focus areas) — desativado
# ==============================================================================

print("\n" + "=" * 60)
print("PIPELINE B — desativado (ver script original para reativar)")
print("=" * 60)
print("\nTodos os pipelines concluídos. Verifique as Tasks no Earth Engine Code Editor.")
