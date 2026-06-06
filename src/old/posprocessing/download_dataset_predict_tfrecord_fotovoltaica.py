#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download Dataset NPY Local - Fotovoltaica (Planet NICFI)
=========================================================
Baixa patches 256×256×8 do mosaico NICFI diretamente para disco local
via ee.data.computePixels(), sem consumir tasks do GEE (limite 3000).

Cada patch salvo em disco:
  <OUTPUT_DIR>/<region_id>/<year>/patch_r<row>_c<col>.npy   →  (256, 256, 8) int16
  <OUTPUT_DIR>/<region_id>/<year>/patch_r<row>_c<col>.json  →  transform + metadados

Canais (axis=-1):
  0  blue   1  green   2  red   3  nir
  4  pvi    5  iia     6  ri    7  evi
FEATURE_BANDS   = ['blue', 'green', 'red', 'nir', 'pvi', 'iia', 'ri', 'evi']
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
import argparse
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

# pathparent = str(Path(os.getcwd()).parents[1])
# sys.path.append(pathparent)
# from configure_account_projects_ee import get_current_account
import collections
collections.Callable = collections.abc.Callable

pathparent = str(Path(os.getcwd()).parents[0])
sys.path.append(pathparent)
# print("parents ", pathparent)
# from configure_account_projects_ee import get_current_account, get_project_from_account
# from gee_tools import *
# projAccount = get_current_account()
# print(f"projetos selecionado >>> {projAccount} <<<")

try:
    ee.Initialize(project= 'mapbiomas-caatinga-cloud04')
    print('The Earth Engine package initialized successfully!')
except ee.EEException as e:
    print('The Earth Engine package failed to initialize!')
except:
    print("Unexpected error:", sys.exc_info()[0])
    raise

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

ASSET_REGIONS = 'projects/mapbiomas-arida/update_02_05_2026_buffer_fotovoltaic_5km'
ASSET_NICFI   = 'projects/planet-nicfi/assets/basemaps/americas'

OUTPUT_DIR    = Path('/dados/dataset_fotovoltaica_npy')   # ← padrão; sobrescrito por --output-dir

YEARS         = list(range(2016, 2026))
PATCH_SIZE    = 256      # pixels
SCALE_M       = 4.77     # metros/pixel — resolução nativa NICFI Planet
STRIDE_PIXELS = 230      # espaçamento entre patches em pixels
CRS           = 'EPSG:4326'  # geográfico; coordenadas em graus

# Escala em graus obtida via projeção GEE (executado após ee.Initialize)
_proj4326  = ee.Projection('EPSG:4326').atScale(SCALE_M).getInfo()
SCALE_DEG  = _proj4326['transform'][0]   # graus/pixel (positivo, eixo x)

STRIDE_DEG     = STRIDE_PIXELS * SCALE_DEG   # graus entre origens de patch
PATCH_SIZE_DEG = PATCH_SIZE    * SCALE_DEG   # extensão angular de um patch

# Controle de fluxo
MAX_RETRIES    = 5     # tentativas por patch antes de desistir
RETRY_WAIT_S   = 15    # espera inicial entre tentativas (dobra a cada retry)
RATE_LIMIT_S   = 1.2   # pausa entre requests — respeita cota GEE (~1 req/s)

# Intervalo de regiões a processar
REGION_INIC = 0
REGION_END  = 90

# Normalização por percentil das bandas brutas do mosaico
dict_percentil = {
    "blue":  [148.69946925122218, 750.681280086019],
    "green": [306.83208478496687, 1166.8854908710632],
    "red":   [181.3067671032011,  1679.1193497398535],
    "nir":   [350.4483690014268,  3933.883473994483],
}

NICFI_BANDS_SRC = ['B', 'G', 'R', 'N']
NICFI_BANDS_DST = ['blue', 'green', 'red', 'nir']
FEATURE_BANDS   = ['blue', 'green', 'red', 'nir', 'pvi', 'iia', 'ri', 'evi']

# ==============================================================================
# 3. FUNÇÕES — IMAGEM
# ==============================================================================

def build_nicfi_mosaic(year: int, geometry) -> ee.Image:
    """
    Mosaico mediana jun–dez do NICFI com bandas normalizadas e índices.

    Filtro jun–dez (não jul) para capturar o mosaico bi-anual Jun–Nov de 2016–2019,
    cujo system:time_start é {year}-06-01 e seria excluído por um filtro a partir de julho.

    Bandas de saída (todas Int16 em [0, 10000]):
      blue, green, red, nir  — normalizadas por percentil
      pvi  = (blue-nir)/(blue+nir+1)         ∈ [-1,  1] → (PVI+1)/2   * 10000
      iia  = (green-4*nir)/(green+4*nir+1)   ∈ [-1,  1] → (IIA+1)/2   * 10000
      ri   = 2.4*(red-green)/(red+green+1)   ∈ [-2.4,2.4] → (RI+2.4)/4.8 * 10000
      evi  = 2.4*(nir-red)/(1+nir+red)       ∈ [-2.4,2.4] → (EVI+2.4)/4.8 * 10000
    """
    mosaic = (ee.ImageCollection(ASSET_NICFI)
              .filterDate(f'{year}-06-01', f'{year + 1}-01-01')
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

    iia = (mosaic.expression(
            'float((green - 4 * nir) / (green + 4 * nir + 1))',
            {'green': mosaic.select('green'), 'nir': mosaic.select('nir')})
           .add(1).divide(2).multiply(10000).toInt16().rename('iia'))

    ri = (mosaic.expression(
            'float(2.4 * (red - green) / (red + green + 1))',
            {'red': mosaic.select('red'), 'green': mosaic.select('green')})
          .add(2.4).divide(4.8).multiply(10000).toInt16().rename('ri'))

    evi = (mosaic.expression(
            'float(2.4 * (nir - red) / (1 + nir + red))',
            {'nir': mosaic.select('nir'), 'red': mosaic.select('red')})
           .add(2.4).divide(4.8).multiply(10000).toInt16().rename('evi'))

    return (scaled.multiply(10000).toInt16()
            .addBands(pvi).addBands(iia).addBands(ri).addBands(evi))


def build_full_stack(year: int, geometry) -> ee.Image:
    """Retorna imagem com 8 bandas espectrais (FEATURE_BANDS) em Int16."""
    mosaic = build_nicfi_mosaic(year, geometry)

    return (mosaic.select(FEATURE_BANDS)
            .clip(geometry)
            .toInt16())

# ==============================================================================
# 4. FUNÇÕES — GRADE DE PATCHES
# ==============================================================================

def get_bbox_4326(geometry) -> tuple:
    """Bounding box da geometria em EPSG:4326 (graus)."""
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

    Alinhamento à grade global de pixels do GEE (EPSG:4326 @ SCALE_DEG):
      x alinhado com floor — garante cobertura a partir da borda esquerda
      y alinhado com ceil  — garante cobertura a partir da borda superior
    """
    start_x = math.floor(minx / SCALE_DEG) * SCALE_DEG
    start_y = math.ceil(maxy  / SCALE_DEG) * SCALE_DEG

    origins, row = [], 0
    y = start_y
    while y > miny:          # patch [y-PATCH_SIZE_DEG, y] ainda intersecta a região
        col, x = 0, start_x
        while x < maxx:      # patch [x, x+PATCH_SIZE_DEG] ainda intersecta a região
            origins.append((row, col, x, y))
            x   += STRIDE_DEG
            col += 1
        y   -= STRIDE_DEG
        row += 1
    return origins

# ==============================================================================
# 5. FUNÇÕES — DOWNLOAD E SALVAMENTO
# ==============================================================================

def _request_patch(image: ee.Image, ox: float, oy: float) -> np.ndarray:
    """
    Baixa um patch 256×256 via computePixels().
    Passa o ee.Image diretamente — o cliente EE serializa internamente.
    Retorna numpy structured array com um campo por banda.
    """
    request = {
        'expression': image,
        'fileFormat': 'NPY',
        'bandIds': FEATURE_BANDS,
        'grid': {
            'dimensions': {'width': PATCH_SIZE, 'height': PATCH_SIZE},
            'affineTransform': {
                'scaleX':     SCALE_DEG,
                'shearX':     0,
                'translateX': ox,
                'shearY':     0,
                'scaleY':    -SCALE_DEG,   # negativo: y decresce para sul
                'translateY': oy,
            },
            'crsCode': CRS,
        },
    }
    raw = ee.data.computePixels(request)
    return np.load(io.BytesIO(raw))


def download_patch_with_retry(image: ee.Image, ox: float, oy: float,
                               label: str) -> np.ndarray:
    """Download com backoff exponencial. Levanta exceção após MAX_RETRIES."""
    for attempt in range(MAX_RETRIES):
        try:
            return _request_patch(image, ox, oy)
        except Exception as exc:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_WAIT_S * (2 ** attempt)
                log.warning(f"    {label} tentativa {attempt + 1}/{MAX_RETRIES}: "
                            f"{exc} — aguardando {wait}s")
                time.sleep(wait)
            else:
                raise


def structured_to_hwc(structured: np.ndarray) -> np.ndarray:
    """Converte numpy structured array (campos = bandas) → (H, W, C) int16."""
    return np.stack([structured[b] for b in FEATURE_BANDS], axis=-1).astype(np.int16)


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
      f"patch_{id_feat}_r{row:04d}_c{col:04d}_{year}.npy"
    """
    fname     = f"patch_{region_id}_r{row:04d}_c{col:04d}_{year}"
    npy_path  = out_dir / f"{fname}.npy"
    json_path = out_dir / f"{fname}.json"

    np.save(npy_path, arr_hwc)

    meta = {
        'region_id': region_id,
        'year':      year,
        'row':       row,
        'col':       col,
        # convenção GDAL: (scale_x, shear_x, origin_x, shear_y, -scale_y, origin_y)
        'transform': [SCALE_DEG, 0.0, ox, 0.0, -SCALE_DEG, oy],
        'crs':       CRS,
        'scale_x':   SCALE_DEG,   # resolução x em graus/pixel
        'scale_y':   SCALE_DEG,   # resolução y em graus/pixel
        'origin_x':  ox,          # borda OESTE do patch (graus, EPSG:4326)
        'origin_y':  oy,          # borda NORTE do patch (graus, EPSG:4326)
        'width':     PATCH_SIZE,
        'height':    PATCH_SIZE,
        'bands':     FEATURE_BANDS,
        'shape':     [PATCH_SIZE, PATCH_SIZE, len(FEATURE_BANDS)],
        'dtype':     'int16',
    }
    json_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    return npy_path

# ==============================================================================
# 6. PIPELINE PRINCIPAL
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Download patches NICFI → NPY local')
    parser.add_argument('--output-dir', type=Path, default=OUTPUT_DIR,
                        help=f'Diretório de saída dos patches (padrão: {OUTPUT_DIR})')
    parser.add_argument('--year_inic', type=int, default=YEARS[0],
                        help=f'Ano inicial do intervalo (padrão: {YEARS[0]})')
    parser.add_argument('--year_end', type=int, default=YEARS[-1],
                        help=f'Ano final do intervalo, inclusivo (padrão: {YEARS[-1]})')
    args = parser.parse_args()
    output_dir = args.output_dir
    years = list(range(args.year_inic, args.year_end + 1))

    log.info("Carregando feature collection de regiões fotovoltaicas...")
    regions_fc  = ee.FeatureCollection(ASSET_REGIONS)
    region_list = (regions_fc
                   .reduceColumns(ee.Reducer.toList(), ['system:index'])
                   .get('list').getInfo())
    total_regions = len(region_list)
    log.info(f"Total de regiões: {total_regions} | Processando [{REGION_INIC}:{REGION_END}]")

    for year in years:
        log.info(f"\n{'='*60}")
        log.info(f"--- Ano {year} ---")

        # Verificação rápida de disponibilidade NICFI para o ano
        num_nicfi = (ee.ImageCollection(ASSET_NICFI)
                     .filterDate(f'{year}-06-01', f'{year + 1}-01-01')
                     .size().getInfo())
        if num_nicfi == 0:
            log.warning(f"  Sem imagens NICFI para {year}. "
                        f"Verifique o acesso ao asset {ASSET_NICFI}. Pulando ano.")
            continue

        for cc, id_feat in enumerate(region_list[REGION_INIC: REGION_END]):
            global_idx = REGION_INIC + cc
            id_safe    = str(id_feat).replace('/', '_').replace(':', '_')

            log.info(f"\n[{global_idx + 1}/{total_regions}] Região: {id_safe}")

            feature = ee.Feature(
                regions_fc.filter(ee.Filter.eq('system:index', id_feat)).first())
            geom = feature.geometry()

            try:
                minx, miny, maxx, maxy = get_bbox_4326(geom)
            except Exception as exc:
                log.error(f"  Erro ao obter bbox: {exc} — pulando.")
                continue

            origins = generate_patch_origins(minx, miny, maxx, maxy)
            log.info(f"  Grade: {len(origins)} patches "
                     f"(stride {STRIDE_PIXELS}px = {STRIDE_DEG:.6f}° | "
                     f"patch {PATCH_SIZE}px = {PATCH_SIZE_DEG:.6f}°)")

            if not origins:
                log.warning("  Nenhum patch gerado. Pulando.")
                continue

            out_dir = output_dir / id_safe / str(year)
            out_dir.mkdir(parents=True, exist_ok=True)

            n_existing = len(list(out_dir.glob('patch_*.npy')))
            log.info(f"  Já no disco: {n_existing}/{len(origins)}")
            if n_existing >= len(origins):
                log.info("  Todos os patches já existem. Pulando.")
                continue

            # Monta imagem UMA VEZ por (região, ano) — passada diretamente ao computePixels
            image = build_full_stack(year, geom)

            patch_iter = tqdm(origins, desc=f"{id_safe}/{year}",
                              unit="patch", disable=not HAS_TQDM)
            downloaded = skipped = failed = 0

            for row, col, ox, oy in patch_iter:
                npy_path = out_dir / f"patch_{id_feat}_r{row:04d}_c{col:04d}_{year}.npy"

                # resume — pula patches já baixados
                if npy_path.exists():
                    skipped += 1
                    continue

                patch_label =f"patch_{id_feat}_r{row:04d}_c{col:04d}_{year}.npy"
                try:
                    structured = download_patch_with_retry(image, ox, oy, patch_label)
                    arr = structured_to_hwc(structured)
                    save_patch(arr, ox, oy, year, id_safe, row, col, out_dir)
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


if __name__ == '__main__':
    main()
