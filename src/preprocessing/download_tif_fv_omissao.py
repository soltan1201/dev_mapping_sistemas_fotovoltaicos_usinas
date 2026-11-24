#!/usr/bin/env python3
"""
Download TIF - pontos de omissão (fv_omissao) para inspeção visual
==================================================================
Base de INSPEÇÃO, sem rótulo: para cada ponto de fv_omissao (painéis que o
predict não detectou, marcados em visualizar_status_predict_rural.js, seção 7)
baixa um recorte quadrado do mosaico NICFI com ~BUFFER_M metros para cada lado
do ponto, direto para o servidor (ee.data.computePixels, sem Drive).

Os TIFs têm as mesmas bandas e escala dos mosaicos rurais
(export_mosaicTIF_GEEtoDrive_rural.py): blue, green, red, pvi, pvpi — int16,
EPSG:3857, 4.77 m. Assim, depois de cada modelo treinado, dá para rodar
direto sobre a pasta:

    python src/processClass/makePredict_fromTIF.py \\
        --model-path <modelo.keras> --input-dir <pasta_destes_tifs>

e comparar como cada modelo se comporta nas omissões.

Nome dos arquivos: omissao_<system_index>_<ano>.tif  (compatível com o
parser <region>_<ano> dos scripts de predict). Um índice CSV
(omissao_index_<ano>.csv) guarda system_index, lon, lat e arquivo.

Requer credencial EE da conta com acesso ao NICFI (solkan1201) no servidor.

Uso:
  python download_tif_fv_omissao.py
  Parâmetros pedidos via input(); Enter aceita o [padrão].
  Especificação: docs/specs/download_dataset_tfrecord_fv_revisao.md
"""

import sys
import csv
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import ee

pathparent = str(Path(__file__).resolve().parents[1])
sys.path.append(pathparent)
from configure_account_projects_ee import get_current_account

# ==============================================================================
# 1. CONFIGURAÇÕES
# ==============================================================================
ASSET_BASE     = 'projects/mapbiomas-arida/energias'
ASSET_NICFI    = 'projects/planet-nicfi/assets/basemaps/americas'
YEAR_ASSET     = 2025
VERSAO_REVISAO = '22_11_2026'
BUFFER_M       = 500          # metros para cada lado do ponto
OUT_DIR        = '~/almacen/db_images/tif_fv_omissao'
WORKERS        = 16
SCALE          = 4.77
CRS            = 'EPSG:3857'
MAX_RETRIES    = 5
R_MERCATOR     = 6378137.0

dict_percentil = {
    'blue':  [100,   800],
    'green': [300,  1200],
    'red':   [176,  1700],
    'nir':   [350,  4000],
}
NICFI_BANDS_SRC = ['B', 'G', 'R', 'N']
NICFI_BANDS_DST = ['blue', 'green', 'red', 'nir']
ALL_BANDS       = ['blue', 'green', 'red', 'pvi', 'pvpi']

# ==============================================================================
# 2. PARÂMETROS DE ENTRADA
# ==============================================================================

def ask(pergunta, padrao, cast=str):
    while True:
        txt = input(f'{pergunta} [{padrao}]: ').strip()
        if not txt:
            return padrao
        try:
            return cast(txt)
        except ValueError:
            print(f'  valor inválido: {txt}')


def ask_sim_nao(pergunta, padrao=False):
    txt = input(f'{pergunta} (s/n) [{"s" if padrao else "n"}]: ').strip().lower()
    return padrao if not txt else txt in ('s', 'sim', 'y', 'yes')


def ask_anos(padrao):
    while True:
        txt = input(f'Anos dos mosaicos (separados por espaço) [{padrao}]: ').strip()
        try:
            return [int(a) for a in txt.split()] if txt else [padrao]
        except ValueError:
            print(f'  valor inválido: {txt}')


print('=' * 60)
print('Parâmetros de entrada (Enter = padrão)')
print('=' * 60)
YEAR_ASSET     = ask('Ano do asset de omissão', YEAR_ASSET, int)
VERSAO_REVISAO = ask('Versão do asset (sufixo do nome)', VERSAO_REVISAO)
YEARS          = ask_anos(YEAR_ASSET)
BUFFER_M       = ask('Buffer em metros para cada lado do ponto', BUFFER_M, float)
OUT_DIR        = Path(ask('Pasta local de saída', OUT_DIR)).expanduser()
WORKERS        = ask('Requisições em paralelo (workers)', WORKERS, int)
DRY_RUN        = ask_sim_nao('Dry-run (só contar pontos)?', False)

ASSET_OMISSAO = f'{ASSET_BASE}/fv_omissao_{YEAR_ASSET}_{VERSAO_REVISAO}'
SIDE_PX       = int(math.ceil(2 * BUFFER_M / SCALE))     # 500 m → 210 px

print(f'\nResumo: asset={ASSET_OMISSAO} | anos={YEARS} | buffer={BUFFER_M} m '
      f'({SIDE_PX}x{SIDE_PX} px) | saída={OUT_DIR} | workers={WORKERS} | dry_run={DRY_RUN}')
if not ask_sim_nao('Confirmar e continuar?', True):
    sys.exit(0)

projAccount = get_current_account()
print(f'Projeto selecionado >>> {projAccount} <<<')
ee.Initialize(project=projAccount, opt_url='https://earthengine-highvolume.googleapis.com')
print('Earth Engine inicializado com sucesso!')

# ==============================================================================
# 3. FUNÇÕES
# ==============================================================================

def build_nicfi_mosaic(year):
    """Igual a export_mosaicTIF_GEEtoDrive_rural.py: 5 bandas int16 [0, 10000]."""
    mosaic = (ee.ImageCollection(ASSET_NICFI)
              .filterDate(f'{year}-07-01', f'{year + 1}-01-01')
              .select(NICFI_BANDS_SRC, NICFI_BANDS_DST)
              .median()
              .toInt16())

    def normaliza(b):
        lo, hi = dict_percentil[b]
        return mosaic.select(b).subtract(lo).divide(hi - lo).clamp(0, 1).rename(b)

    scaled = ee.Image.cat([normaliza(b) for b in NICFI_BANDS_DST])
    pvi = (mosaic.expression(
               'float(BLUE - NIR) / float(BLUE + NIR + 1)',
               {'BLUE': mosaic.select('blue'), 'NIR': mosaic.select('nir')})
           .add(1).divide(2).multiply(10000).toInt16().rename('pvi'))
    pvpi = (mosaic.expression(
                'float((green - blue) / (green + blue))',
                {'green': mosaic.select('green'), 'blue': mosaic.select('blue')})
            .add(1).divide(2).multiply(10000).toInt16().rename('pvpi'))
    return (scaled.multiply(10000).toInt16().select(['blue', 'green', 'red'])
            .addBands(pvi).addBands(pvpi).unmask(0).toInt16())


def lonlat_to_3857(lon, lat):
    x = R_MERCATOR * np.radians(lon)
    y = R_MERCATOR * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
    return x, y


def fetch_tif(image, lon, lat, path):
    """Recorte SIDE_PX x SIDE_PX centrado no ponto, gravado como GeoTIFF."""
    x, y = lonlat_to_3857(lon, lat)
    x0 = math.floor((x - BUFFER_M) / SCALE) * SCALE     # alinhado à grade de pixels
    y0 = math.ceil((y + BUFFER_M) / SCALE) * SCALE
    request = {
        'expression': image,
        'fileFormat': 'GEO_TIFF',
        'bandIds':    ALL_BANDS,
        'grid': {
            'dimensions': {'width': SIDE_PX, 'height': SIDE_PX},
            'affineTransform': {
                'scaleX': SCALE, 'shearX': 0, 'translateX': x0,
                'shearY': 0, 'scaleY': -SCALE, 'translateY': y0,
            },
            'crsCode': CRS,
        },
    }
    for attempt in range(MAX_RETRIES):
        try:
            raw = ee.data.computePixels(request)
            tmp = path.with_suffix('.part')
            tmp.write_bytes(raw)
            tmp.rename(path)
            return True
        except Exception as exc:
            if attempt == MAX_RETRIES - 1:
                print(f'    ❌ {path.name}: {exc}')
                return False
            time.sleep(2 ** attempt)

# ==============================================================================
# 4. PIPELINE
# ==============================================================================
try:
    ee.data.getAsset(ASSET_OMISSAO)
except Exception:
    print(f'❌ Asset não encontrado: {ASSET_OMISSAO}')
    sys.exit(1)

fc = ee.FeatureCollection(ASSET_OMISSAO)
# garante ponto em lon/lat (se vier polígono, usa o centróide)
fc = fc.map(lambda f: ee.Feature(f.geometry().centroid(1).transform('EPSG:4326', 1),
                                 {'sid': f.get('system:index')}))
feats  = fc.getInfo()['features']       # até 5000 pontos
pontos = [(f['properties']['sid'], *f['geometry']['coordinates']) for f in feats]
print(f'\nPontos de omissão: {len(pontos)}')

if DRY_RUN or not pontos:
    sys.exit(0)

OUT_DIR.mkdir(parents=True, exist_ok=True)

for year in YEARS:
    image = build_nicfi_mosaic(year)
    jobs  = []
    for sid, lon, lat in pontos:
        path = OUT_DIR / f'omissao_{sid}_{year}.tif'
        jobs.append((sid, lon, lat, path))

    pend = [j for j in jobs if not j[3].exists()]
    print(f'\nAno {year}: {len(jobs)} pontos, {len(jobs) - len(pend)} já baixados, '
          f'{len(pend)} pendentes')

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        ok = list(ex.map(lambda j: fetch_tif(image, j[1], j[2], j[3]), pend))
    print(f'  baixados: {sum(ok)}  |  erros: {len(ok) - sum(ok)}')

    index_path = OUT_DIR / f'omissao_index_{year}.csv'
    with open(index_path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['system_index', 'lon', 'lat', 'year', 'arquivo', 'existe'])
        for sid, lon, lat, path in jobs:
            w.writerow([sid, lon, lat, year, path.name, path.exists()])
    print(f'  índice → {index_path}')

print('\nConcluído.')
