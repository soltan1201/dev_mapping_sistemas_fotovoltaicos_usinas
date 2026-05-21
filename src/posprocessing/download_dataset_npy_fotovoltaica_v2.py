#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download Dataset NPY Local - Fotovoltaica (Planet NICFI)
=========================================================
Baixa patches 256×256×9 do mosaico NICFI diretamente para disco local
via ee.data.computePixels(), sem consumir tasks do GEE (limite 3000).

Cada patch salvo em disco:
  <OUTPUT_DIR>/<region_id>/<year>/patch_r<row>_c<col>.npy   →  (256, 256, 9) int16
  <OUTPUT_DIR>/<region_id>/<year>/patch_r<row>_c<col>.json  →  transform + metadados

Canais (axis=-1):
  0  blue   1  green   2  red   3  nir
  4  pvi    5  iia     6  ri    7  evi   8  label

Transform salvo no JSON é o necessário para georreferência do predict:
  [scale_x, 0, origin_x, 0, -scale_y, origin_y]   (convenção GDAL/rasterio)
"""

import sys
import os
import io
import math
import json
import time
import logging
import numpy as np
import ee
from pathlib import Path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(it, **kw):
        return it

pathparent = str(Path(os.getcwd()).parents[1])
sys.path.append(pathparent)
from configure_account_projects_ee import get_current_account

# ==============================================================================
# 1. LOGGING
# ==============================================================================

LOG_FILE = Path('export_npy.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
    ],
)
log = logging.getLogger(__name__)

# ==============================================================================
# 2. CONFIGURAÇÕES
# ==============================================================================

ASSET_REGIONS = "projects/mapbiomas-arida/energias/shp_area_fotovoltaic_samples_update_16_05_2026"
ASSET_LABEL   = 'projects/mapbiomas-workspace/AMOSTRAS/col10/CAATINGA/solar-panel-br-30m_2016_2024_v2'
ASSET_NICFI   = 'projects/planet-nicfi/assets/basemaps/americas'

OUTPUT_DIR    = Path('/dados/dataset_fotovoltaica_npy')   # ← ajuste para o servidor

YEARS         = [2019, 2022, 2024]
PATCH_SIZE    = 256      # pixels
SCALE         = 4.77     # metros/pixel — resolução nativa NICFI Planet
STRIDE_PIXELS = 200      # espaçamento entre patches em pixels
CRS           = 'EPSG:3857'  # projeção métrica; garante pixels quadrados e uniformes

# Controle de fluxo
MAX_RETRIES    = 5     # tentativas por patch antes de desistir
RETRY_WAIT_S   = 15    # espera inicial entre tentativas (dobra a cada retry)
RATE_LIMIT_S   = 1.2   # pausa entre requests — respeita cota GEE (~1 req/s)

# Intervalo de regiões a processar
REGION_INIC = 0
REGION_END  = 50

# Normalização por percentil das bandas brutas do mosaico
dict_percentil = {
    "blue":  [100,   800],
    "green": [300,  1200],
    "red":   [176,  1700],
    "nir":   [350,  4000]
}

NICFI_BANDS_SRC = ['B', 'G', 'R', 'N']
NICFI_BANDS_DST = ['blue', 'green', 'red', 'nir']
# FEATURE_BANDS   = ['blue', 'green', 'red',  'pvi', 'ndwi', 'ri', 'evi']
FEATURE_BANDS   = ['blue', 'green', 'red', 'pvi', 'pvpi']
ALL_BANDS       = FEATURE_BANDS + ['label']   # 5 canais → shape (256, 256, 5)

STRIDE_M     = STRIDE_PIXELS * SCALE    # metros entre origens de patch
PATCH_SIZE_M = PATCH_SIZE * SCALE       # extensão métrica de um patch

# ==============================================================================
# 3. INICIALIZAÇÃO EE
# ==============================================================================

projAccount = get_current_account()
log.info(f"Projeto selecionado: {projAccount}")

try:
    ee.Initialize(project=projAccount)
    log.info('Earth Engine inicializado com sucesso.')
except Exception as e:
    log.error(f"Erro de inicialização: {e}")
    raise

# ==============================================================================
# 4. FUNÇÕES — IMAGEM
# ==============================================================================

def build_nicfi_mosaic(year: int, geometry) -> ee.Image:
    """
    Mosaico mediana jul–dez do NICFI com bandas normalizadas e índices.

    Bandas de saída (todas Int16 em [0, 10000]):
      blue, green, red, nir  — normalizadas por percentil
      pvi  = (blue-nir)/(blue+nir+1)         ∈ [-1,  1] → (PVI+1)/2   * 10000
      iia  = (green-4*nir)/(green+4*nir+1)   ∈ [-1,  1] → (IIA+1)/2   * 10000
      ri   = 2.4*(red-green)/(red+green+1)   ∈ [-2.4,2.4] → (RI+2.4)/4.8 * 10000
      evi  = 2.4*(nir-red)/(1+nir+red)       ∈ [-2.4,2.4] → (EVI+2.4)/4.8 * 10000
    """
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
            .addBands(pvi).addBands(pvpi))


def build_full_stack(year: int, geometry) -> ee.Image:
    """Empilha 8 bandas espectrais + rótulo binarizado → 9 canais Int16."""
    mosaic = build_nicfi_mosaic(year, geometry)
    label  = (ee.Image(ASSET_LABEL)
              .select(f'Panel_{year}')
              .gte(1).rename('label').toByte())
    return (mosaic.select(FEATURE_BANDS)
            .addBands(label)
            .clip(geometry)
            .toInt16())

# ==============================================================================
# 5. FUNÇÕES — GRADE DE PATCHES
# ==============================================================================

def get_bbox_3857(geometry) -> tuple:
    """Bounding box da geometria em EPSG:3857 (metros)."""
    bounds = geometry.bounds(1, ee.Projection(CRS)).getInfo()
    coords = bounds['coordinates'][0]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return min(xs), min(ys), max(xs), max(ys)


def generate_patch_origins(minx: float, miny: float,
                            maxx: float, maxy: float) -> list:
    """
    Gera lista de (row, col, origin_x, origin_y) para patches que cobrem o bbox.

    origin_x = borda oeste do patch (translateX na transformação afim)
    origin_y = borda norte do patch (translateY na transformação afim)

    Alinhamento à grade global de pixels do GEE (EPSG:3857 @ SCALE):
      x alinhado com floor — garante cobertura a partir da borda esquerda
      y alinhado com ceil  — garante cobertura a partir da borda superior
    """
    start_x = math.floor(minx / SCALE) * SCALE
    start_y = math.ceil(maxy  / SCALE) * SCALE

    origins, row = [], 0
    y = start_y
    while y > miny:          # patch [y-PATCH_SIZE_M, y] ainda intersecta a região
        col, x = 0, start_x
        while x < maxx:      # patch [x, x+PATCH_SIZE_M] ainda intersecta a região
            origins.append((row, col, x, y))
            x   += STRIDE_M
            col += 1
        y   -= STRIDE_M
        row += 1
    return origins

# ==============================================================================
# 6. FUNÇÕES — DOWNLOAD E SALVAMENTO
# ==============================================================================

def _request_patch(image_serialized: dict,
                   ox: float, oy: float) -> np.ndarray:
    """
    Baixa um patch 256×256 via computePixels().
    Retorna numpy structured array com um campo por banda.
    """
    request = {
        'expression': image_serialized,
        'fileFormat': 'NPY',
        'grid': {
            'dimensions': {'width': PATCH_SIZE, 'height': PATCH_SIZE},
            'affineTransform': {
                'scaleX':     SCALE,
                'shearX':     0,
                'translateX': ox,
                'shearY':     0,
                'scaleY':    -SCALE,   # negativo: y decresce para sul
                'translateY': oy,
            },
            'crsCode': CRS,
        },
    }
    raw = ee.data.computePixels(request)
    return np.load(io.BytesIO(raw))


def download_patch_with_retry(image_serialized, ox: float, oy: float,
                               label: str) -> np.ndarray:
    """Download com backoff exponencial. Levanta exceção após MAX_RETRIES."""
    for attempt in range(MAX_RETRIES):
        try:
            return _request_patch(image_serialized, ox, oy)
        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_WAIT_S * (2 ** attempt)
                log.warning(f"    {label} tentativa {attempt + 1}/{MAX_RETRIES}: "
                            f"{exc} — aguardando {wait}s")
                time.sleep(wait)
            else:
                raise


def structured_to_hwc(structured: np.ndarray) -> np.ndarray:
    """
    Converte numpy structured array (campos = bandas) → (H, W, C) int16.
    C = len(ALL_BANDS) = 9
    """
    return np.stack([structured[b] for b in ALL_BANDS], axis=-1).astype(np.int16)


def save_patch(arr_hwc: np.ndarray,
               ox: float, oy: float,
               year: int, region_id: str,
               row: int, col: int,
               out_dir: Path) -> Path:
    """
    Salva patch como .npy (H, W, C) int16 e metadado .json.

    O campo 'transform' no JSON segue a convenção GDAL/rasterio:
      (scale_x, shear_x, origin_x, shear_y, -scale_y, origin_y)
    Permite reconstrução direta com:
      rasterio.transform.from_gdal(*meta['transform'])
    """
    fname     = f"patch_r{row:04d}_c{col:04d}"
    npy_path  = out_dir / f"{fname}.npy"
    json_path = out_dir / f"{fname}.json"

    np.save(npy_path, arr_hwc)

    meta = {
        'region_id': region_id,
        'year':      year,
        'row':       row,
        'col':       col,
        # convenção GDAL: (scale_x, shear_x, origin_x, shear_y, -scale_y, origin_y)
        'transform': [SCALE, 0.0, ox, 0.0, -SCALE, oy],
        'crs':       CRS,
        'scale_x':   SCALE,   # resolução x em metros/pixel
        'scale_y':   SCALE,   # resolução y em metros/pixel
        'origin_x':  ox,      # borda OESTE do patch (metros, EPSG:3857)
        'origin_y':  oy,      # borda NORTE do patch (metros, EPSG:3857)
        'width':     PATCH_SIZE,
        'height':    PATCH_SIZE,
        'bands':     ALL_BANDS,
        'shape':     [PATCH_SIZE, PATCH_SIZE, len(ALL_BANDS)],
        'dtype':     'int16',
    }
    json_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    return npy_path

# ==============================================================================
# 7. PIPELINE PRINCIPAL
# ==============================================================================

log.info("Carregando feature collection de regiões fotovoltaicas...")
regions_fc    = ee.FeatureCollection(ASSET_REGIONS)
region_list   = regions_fc.toList(regions_fc.size())
total_regions = regions_fc.size().getInfo()


log.info(f"Total de regiões: {total_regions} | Processando [{REGION_INIC}:{REGION_END}]")

for cc in range(min(REGION_END - REGION_INIC, total_regions - REGION_INIC)):
    global_idx = REGION_INIC + cc
    feature    = ee.Feature(region_list.get(global_idx))
    geom       = feature.geometry()

    feat_id      = feature.get('system:index').getInfo() or f'{global_idx:04d}'
    feat_id_safe = str(feat_id).replace('/', '_').replace(':', '_')

    log.info(f"\n{'='*60}")
    log.info(f"[{global_idx + 1}/{total_regions}] Região: {feat_id_safe}")

    try:
        minx, miny, maxx, maxy = get_bbox_3857(geom)
    except Exception as exc:
        log.error(f"  Erro ao obter bbox: {exc} — pulando.")
        continue

    origins = generate_patch_origins(minx, miny, maxx, maxy)
    log.info(f"  Grade: {len(origins)} patches "
             f"(stride {STRIDE_PIXELS}px = {STRIDE_M:.0f}m | "
             f"patch {PATCH_SIZE}px = {PATCH_SIZE_M:.0f}m)")

    if not origins:
        log.warning("  Nenhum patch gerado. Pulando.")
        continue

    for year in YEARS:
        log.info(f"\n  --- Ano {year} ---")

        out_dir = OUTPUT_DIR / feat_id_safe / str(year)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Monta imagem (8 features + label) e serializa UMA VEZ por (região, ano)
        image            = build_full_stack(year, geom)
        image_serialized = ee.serializer.encode(image)

        n_existing = len(list(out_dir.glob('patch_*.npy')))
        log.info(f"  Já no disco: {n_existing}/{len(origins)}")

        patch_iter = tqdm(origins, desc=f"{feat_id_safe}/{year}",
                          unit="patch", disable=not HAS_TQDM)
        downloaded = skipped = failed = 0

        for row, col, ox, oy in patch_iter:
            npy_path = out_dir / f"patch_r{row:04d}_c{col:04d}.npy"

            # resume — pula patches já baixados
            if npy_path.exists():
                skipped += 1
                continue

            patch_label = f"r{row:04d}_c{col:04d}"
            try:
                structured = download_patch_with_retry(
                    image_serialized, ox, oy, patch_label)
                arr = structured_to_hwc(structured)
                save_patch(arr, ox, oy, year, feat_id_safe, row, col, out_dir)
                downloaded += 1
            except Exception as exc:
                log.error(f"  Falha definitiva em {patch_label}: {exc}")
                failed += 1
            finally:
                time.sleep(RATE_LIMIT_S)

        log.info(f"  Baixados: {downloaded} | "
                 f"Pulados (já existiam): {skipped} | "
                 f"Falhas: {failed}")

log.info(f"\nExportação concluída. Log salvo em: {LOG_FILE.resolve()}")
