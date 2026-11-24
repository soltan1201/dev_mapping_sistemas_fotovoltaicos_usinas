#!/usr/bin/env python3
"""
Download Dataset TFRecord - Revisão do predict (fv_confirm / fv_false)
======================================================================
Gera uma nova base de TFRecords a partir dos polígonos de revisão
desenhados no GEE (visualizar_status_predict_rural.js, seção 7):

  fv_confirm  → predict CORRETO na área   → label = predict
  fv_false    → predict ERRADO na área    → label = 0 (tudo em branco no polígono)

Os pontos de omissão (fv_omissao) NÃO entram aqui: vão para a base de inspeção
em TIF, gerada por download_tif_fv_omissao.py.

Rótulo de cada patch:
  1. label = predição FV do ano, a MESMA camada "Predição <ano>" de
     visualizar_status_predict_rural.js:
       ImageCollection(ASSET_PREDICOES [+ ASSET_PRED_RURAL])
         .filter(year).mosaic() > 0
  2. fv_false vira máscara: manchas de FV do predict dentro dela são falsos
     positivos e são apagadas (0) — antes de amostrar, nas duas camadas
  3. depois os patches são amostrados dentro de cada polígono
  fora dos polígonos   → depende de LABEL_OUTSIDE:
        'pred' : mantém o predict (padrão — patch inteiro tem rótulo)
        'zero' : zera tudo fora de fv_confirm

Mesmo formato das bases anteriores (download_dataset_tfrecord_fotovoltaica_precomp.py):
  bandas ['blue', 'green', 'red', 'pvi', 'pvpi'] + 'label', patch 256x256, 4.77 m.

Pontos de patch (cada polígono tratado individualmente, sem união nem buffer):
  - fv_false  : N_PONTOS_FALSE (5) pontos aleatórios dentro do polígono
  - fv_confirm: área estratificada em N_CLASSES_CONFIRM (6) classes por quantil;
                classe 0 (menores) = 1 patch no centro, depois 2, 4, 8, 16, 32
                (centro + pontos aleatórios dentro do polígono)

Destino (pergunta "Destino"):
  drive : Export.table.toDrive (tasks GEE) → EXPORT_DRIVE_FOLDER
  local : baixa cada patch direto para o servidor via ee.data.computePixels
          (sem Drive, sem tasks) e grava *.tfrecord.gz no mesmo formato do
          export do GEE: cada banda = float_list 257x257 (kernel 128).
          Precisa de credencial EE da conta com acesso ao NICFI no servidor:
            earthengine authenticate   (logar com a conta solkan1201)
            echo solkan1201 > ~/.config/earthengine/current_user.txt

Uso:
  python download_dataset_tfrecord_fv_revisao.py
  Os parâmetros são pedidos via input(); Enter aceita o valor [padrão].
  Especificação completa: docs/specs/download_dataset_tfrecord_fv_revisao.md
"""

import sys
import os
import csv
import math
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import ee

pathparent = str(Path(__file__).resolve().parents[1])
sys.path.append(pathparent)
from configure_account_projects_ee import get_current_account

# ==============================================================================
# 1. CONFIGURAÇÕES
# ==============================================================================
VERSAO_REVISAO = '22_11_2026'
YEAR           = 2025

ASSET_BASE     = 'projects/mapbiomas-arida/energias'
TIPOS          = ['confirm', 'false']
OUT_DIR        = '~/almacen/db_images/tfrecord_fv_revisao'
WORKERS        = 16

# ── Camada de predict (rótulo) ────────────────────────────────────────────────
# Deve ser IGUAL a asset_predicoes / ASSET_PRED_RURAL de
# src/show_colect_data/visualizar_status_predict_rural.js
ASSET_PREDICOES  = 'projects/geo-data-s/assets/fotovoltaica/usinas_br_gc'
ASSET_PRED_RURAL = None   # ex.: 'projects/geo-data-s/assets/fotovoltaica/rural_br'
LABEL_OUTSIDE    = 'pred' # 'pred' | 'zero'

ASSET_NICFI = 'projects/planet-nicfi/assets/basemaps/americas'

# ── Saída ─────────────────────────────────────────────────────────────────────
EXPORT_DRIVE_FOLDER = 'DS_FV_NICFI_TFR_REVISAO'
TASK_PREFIX         = 'tfrecord_fvrev_'

# ── Patch / grade ─────────────────────────────────────────────────────────────
PATCH_SIZE       = 256
SCALE            = 4.77
MAX_PATCHES_FILE = 50
SKIP_COMPLETED   = True

# ── Amostragem por polígono ───────────────────────────────────────────────────
N_PONTOS_FALSE    = 5   # fv_false: pontos aleatórios por polígono
N_CLASSES_CONFIRM = 6   # fv_confirm: classes de área (quantis) → 1, 2, 4, 8, 16, 32 pontos

dict_percentil = {
    'blue':  [100,   800],
    'green': [300,  1200],
    'red':   [176,  1700],
    'nir':   [350,  4000],
}
NICFI_BANDS_SRC = ['B', 'G', 'R', 'N']
NICFI_BANDS_DST = ['blue', 'green', 'red', 'nir']
ALL_BANDS       = ['blue', 'green', 'red', 'pvi', 'pvpi']
SELECTORS       = ALL_BANDS + ['label']

# ==============================================================================
# 2. ARGUMENTOS / INIT
# ==============================================================================
def ask(pergunta, padrao, cast=str, opcoes=None):
    """input() com valor padrão (Enter) e validação simples."""
    while True:
        txt = input(f'{pergunta} [{padrao}]: ').strip()
        if not txt:
            return padrao
        try:
            val = cast(txt)
        except ValueError:
            print(f'  valor inválido: {txt}')
            continue
        if opcoes and val not in opcoes:
            print(f'  opções válidas: {opcoes}')
            continue
        return val


def ask_sim_nao(pergunta, padrao=False):
    txt = input(f'{pergunta} (s/n) [{"s" if padrao else "n"}]: ').strip().lower()
    return padrao if not txt else txt in ('s', 'sim', 'y', 'yes')


def ask_tipos(padrao):
    while True:
        txt = input(f'Tipos de polígono (separados por espaço) [{" ".join(padrao)}]: ').strip()
        tipos = txt.split() if txt else padrao
        invalid = [t for t in tipos if t not in TIPOS]
        if not invalid:
            return tipos
        print(f'  inválidos: {invalid} — opções: {TIPOS}')


print('=' * 60)
print('Parâmetros de entrada (Enter = padrão)')
print('=' * 60)
_destino       = ask('Destino (drive | local)', 'drive', opcoes=['drive', 'local'])
YEAR           = ask('Ano', YEAR, int)
VERSAO_REVISAO = ask('Versão dos assets de revisão (sufixo do nome)', VERSAO_REVISAO)
_tipos         = ask_tipos(TIPOS)
LABEL_OUTSIDE  = ask('Rótulo fora dos polígonos (pred | zero)', LABEL_OUTSIDE,
                     opcoes=['pred', 'zero'])
_out_dir, _workers = OUT_DIR, WORKERS
if _destino == 'local':
    _out_dir = ask('Pasta local de saída', OUT_DIR)
    _workers = ask('Requisições em paralelo (workers)', WORKERS, int)
_dry_run = ask_sim_nao('Dry-run (só contar, não exportar/baixar)?', False)

_args = SimpleNamespace(tipos=_tipos, dry_run=_dry_run, destino=_destino,
                        out_dir=_out_dir, workers=_workers)
LOCAL = _destino == 'local'

ASSETS_REVISAO = {t: f'{ASSET_BASE}/fv_{t}_{YEAR}_{VERSAO_REVISAO}' for t in TIPOS}
print(f'\nResumo: {vars(_args)} | ano={YEAR} | versão={VERSAO_REVISAO} '
      f'| label_outside={LABEL_OUTSIDE}')
if not ask_sim_nao('Confirmar e continuar?', True):
    sys.exit(0)

# projAccount = get_current_account()   # conta (credencial) = solkan1201
projAccount = 'mapbiomas-brazil'      # projeto de cota/cobrança (solkan1201 tem permissão)
print(f'Projeto selecionado >>> {projAccount} <<<')
if LOCAL:
    # endpoint de alto volume: recomendado para muitas requisições computePixels
    ee.Initialize(project=projAccount,
                  opt_url='https://earthengine-highvolume.googleapis.com')
else:
    ee.Initialize(project=projAccount)
print('Earth Engine inicializado com sucesso!')


def get_completed_tasks(prefix):
    try:
        return {t['description'] for t in ee.data.getTaskList()
                if t.get('state') == 'COMPLETED'
                and t.get('description', '').startswith(prefix)}
    except Exception as e:
        print(f'  Aviso: não foi possível buscar tasks concluídas: {e}')
        return set()


_completed = get_completed_tasks(TASK_PREFIX) if (SKIP_COMPLETED and not LOCAL) else set()

# ==============================================================================
# 3. IMAGENS
# ==============================================================================

def build_nicfi_mosaic(year, geometry):
    """Mosaico mediana jul–dez NICFI, 5 bandas (igual ao *_precomp.py)."""
    col = ee.ImageCollection(ASSET_NICFI).filterDate(f'{year}-07-01', f'{year + 1}-01-01')
    if geometry is not None:
        col = col.filterBounds(geometry)
    mosaic = (col.select(NICFI_BANDS_SRC, NICFI_BANDS_DST)
              .median()
              .toInt16())

    def normaliza(b):
        lo, hi = dict_percentil[b]
        return (mosaic.select(b).subtract(lo).divide(hi - lo)
                .clamp(0, 1).rename(b))

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
            .addBands(pvi).addBands(pvpi))


def pred_collection(year):
    """Mesma camada 'Predição <ano>' do visualizar_status_predict_rural.js."""
    col = ee.ImageCollection(ASSET_PREDICOES)
    if ASSET_PRED_RURAL:
        col = col.merge(ee.ImageCollection(ASSET_PRED_RURAL))
    return col.filter(ee.Filter.eq('year', year))


def build_label(pred_bin, fc_confirm, fc_false):
    """1) fv_false vira máscara: toda mancha de FV do predict dentro dela é falso
    positivo e é apagada (label 0) — vale para os patches das DUAS camadas.
    2) opcionalmente 0 fora de fv_confirm (LABEL_OUTSIDE='zero')."""
    label = pred_bin
    if LABEL_OUTSIDE == 'zero':
        label = label.multiply(ee.Image(0).paint(fc_confirm, 1))
    mascara_false = ee.Image(0).paint(fc_false, 1)
    label = label.where(mascara_false.eq(1), 0)
    return label.rename('label').toByte()


def build_stack(year, fc_confirm, fc_false):
    """5 bandas + label, int16. Sem clip e sem buffer: o patch pedido (grid do
    computePixels ou kernel do neighborhoodToArray) já limita a área."""
    mosaic = build_nicfi_mosaic(year, None)
    pred   = pred_collection(year).mosaic().gt(0).unmask(0)
    label  = build_label(pred, fc_confirm, fc_false)
    return (mosaic.select(ALL_BANDS).unmask(0)
            .addBands(label.unmask(0))
            .toInt16())

# ==============================================================================
# 4. PONTOS (cada polígono tratado individualmente, calculado no Python)
# ==============================================================================
R_MERCATOR = 6378137.0


def lonlat_to_3857(lon, lat):
    x = R_MERCATOR * np.radians(lon)
    y = R_MERCATOR * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
    return x, y


def mercator_to_lonlat(x, y):
    lon = np.degrees(x / R_MERCATOR)
    lat = np.degrees(2 * np.arctan(np.exp(y / R_MERCATOR)) - np.pi / 2)
    return float(lon), float(lat)


def _polys_3857(geom):
    """Lista de polígonos; cada polígono = lista de anéis (arrays Nx2 em EPSG:3857)."""
    if geom['type'] == 'Polygon':
        polys = [geom['coordinates']]
    elif geom['type'] == 'MultiPolygon':
        polys = geom['coordinates']
    else:
        return []
    out = []
    for poly in polys:
        rings = []
        for ring in poly:
            arr = np.asarray(ring, dtype=float)
            x, y = lonlat_to_3857(arr[:, 0], arr[:, 1])
            rings.append(np.column_stack([x, y]))
        out.append(rings)
    return out


def _in_ring(x, y, ring):
    xs, ys = ring[:, 0], ring[:, 1]
    xj, yj = np.roll(xs, 1), np.roll(ys, 1)
    cruza = ((ys > y) != (yj > y)) & (x < (xj - xs) * (y - ys) / (yj - ys + 1e-12) + xs)
    return bool(np.count_nonzero(cruza) % 2)


def _in_polys(x, y, polys):
    return any(_in_ring(x, y, p[0]) and not any(_in_ring(x, y, h) for h in p[1:])
               for p in polys)


def _area_ha(polys):
    """Área aproximada (shoelace em 3857 corrigido por cos²(lat))."""
    total, ys = 0.0, []
    for p in polys:
        for k, ring in enumerate(p):
            x, y = ring[:, 0], ring[:, 1]
            a = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            total += a if k == 0 else -a
            ys.append(y.mean())
    lat = mercator_to_lonlat(0, float(np.mean(ys)))[1]
    return total * math.cos(math.radians(lat)) ** 2 / 1e4


def _random_inside(polys, n, rng):
    """n pontos aleatórios uniformes dentro do polígono (rejeição no bbox)."""
    allxy = np.vstack([r for p in polys for r in p])
    (x0, y0), (x1, y1) = allxy.min(0), allxy.max(0)
    pts, tentativas = [], 0
    while len(pts) < n and tentativas < 1000 * n:
        x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
        tentativas += 1
        if _in_polys(x, y, polys):
            pts.append((x, y))
    return pts


def _centro(polys):
    allxy = np.vstack([r for p in polys for r in p])
    cx, cy = (allxy.min(0) + allxy.max(0)) / 2
    return (cx, cy) if _in_polys(cx, cy, polys) else None


def pontos_por_poligono(feats, tipo, seed):
    """Cada polígono gera seus próprios patches (sem união, sem buffer).
      false   : N_PONTOS_FALSE pontos aleatórios dentro do polígono.
      confirm : área estratificada em N_CLASSES_CONFIRM classes (quantis);
                classe 0 (menores) = 1 ponto no centro, depois 2, 4, 8, ...
                (centro + aleatórios dentro do polígono).
    Retorna lista de dicts: sid, classe, area_ha, n, lon, lat."""
    rng  = np.random.default_rng(seed)
    info = []
    for f in feats:
        polys = _polys_3857(f['geometry'])
        if polys:
            info.append((f.get('id', ''), polys, _area_ha(polys)))
    if not info:
        return [], None

    areas = np.array([a for _, _, a in info])
    if tipo == 'confirm':
        cortes = np.quantile(areas, np.linspace(0, 1, N_CLASSES_CONFIRM + 1)[1:-1])
        classes = np.searchsorted(cortes, areas, side='right')
    else:
        cortes, classes = None, np.zeros(len(info), dtype=int)

    pts = []
    for (sid, polys, area), classe in zip(info, classes):
        if tipo == 'confirm':
            n = 2 ** int(classe)
            xy = [c for c in [_centro(polys)] if c is not None]
            xy += _random_inside(polys, n - len(xy), rng)
        else:
            n  = N_PONTOS_FALSE
            xy = _random_inside(polys, n, rng)
        for x, y in xy:
            lon, lat = mercator_to_lonlat(x, y)
            pts.append({'sid': sid, 'classe': int(classe), 'area_ha': round(area, 2),
                        'n': n, 'lon': lon, 'lat': lat})

    order = rng.permutation(len(pts))          # mistura polígonos entre os shards
    return [pts[i] for i in order], (cortes, classes, areas)


def resumo_estratos(tipo, estratos, n_pts):
    cortes, classes, areas = estratos
    if tipo != 'confirm':
        print(f'  {len(areas)} polígonos × {N_PONTOS_FALSE} pontos aleatórios = {n_pts} pontos')
        return
    lim = [areas.min(), *cortes, areas.max()]
    print('  classe  área (ha)              polígonos  pontos/polígono')
    for k in range(N_CLASSES_CONFIRM):
        print(f'  {k:>6}  {lim[k]:>9.2f} – {lim[k + 1]:<9.2f}  '
              f'{int((classes == k).sum()):>9}  {2 ** k:>15}')
    print(f'  total de pontos: {n_pts}')

# ==============================================================================
# 5. EXPORT
# ==============================================================================

def export_shard(patches_array, points, fname):
    task = ee.batch.Export.table.toDrive(
        collection=patches_array.sampleRegions(
            collection=points, scale=SCALE, geometries=False, tileScale=16),
        description=fname,
        folder=EXPORT_DRIVE_FOLDER,
        fileFormat='TFRecord',
        selectors=SELECTORS,
    )
    task.start()


# ==============================================================================
# 5b. DOWNLOAD LOCAL (computePixels → TFRecord GZIP)
# ==============================================================================
RAW_PATCH   = PATCH_SIZE + 1          # 257: mesmo tamanho do neighborhoodToArray(128)
HALF        = PATCH_SIZE // 2         # 128
MAX_RETRIES = 5


def fetch_patch(stack, lon, lat):
    """Patch 257x257 centrado no pixel do ponto (EPSG:3857 @ SCALE)."""
    x, y = lonlat_to_3857(lon, lat)
    col  = np.floor(x / SCALE)            # coluna/linha do pixel que contém o ponto
    row  = np.floor(y / SCALE)
    request = {
        'expression': stack,
        'fileFormat': 'NPY',
        'bandIds':    SELECTORS,
        'grid': {
            'dimensions': {'width': RAW_PATCH, 'height': RAW_PATCH},
            'affineTransform': {
                'scaleX': SCALE, 'shearX': 0, 'translateX': (col - HALF) * SCALE,
                'shearY': 0, 'scaleY': -SCALE, 'translateY': (row + 1 + HALF) * SCALE,
            },
            'crsCode': 'EPSG:3857',
        },
    }
    for attempt in range(MAX_RETRIES):
        try:
            raw = ee.data.computePixels(request)
            arr = np.load(io.BytesIO(raw))
            return {b: arr[b].astype(np.float32) for b in SELECTORS}
        except Exception as exc:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = 2 ** attempt
            print(f'      retry {attempt + 1} ({exc}); aguardando {wait}s')
            time.sleep(wait)


def to_example(tf, patch):
    feats = {b: tf.train.Feature(float_list=tf.train.FloatList(value=patch[b].ravel()))
             for b in SELECTORS}
    return tf.train.Example(features=tf.train.Features(feature=feats)).SerializeToString()


def download_local(stack, coords, tipo, out_dir):
    import tensorflow as tf   # só no modo local

    n_shards = max(1, math.ceil(len(coords) / MAX_PATCHES_FILE))
    out_dir.mkdir(parents=True, exist_ok=True)
    opts = tf.io.TFRecordOptions(compression_type='GZIP')
    print(f'  saída: {out_dir}')

    for i in range(n_shards):
        path = out_dir / f'{TASK_PREFIX}{tipo}_{YEAR}_{VERSAO_REVISAO}_part{i:03d}.tfrecord.gz'
        if path.exists():
            print(f'    {path.name}: já existe, pulando.')
            continue
        chunk = coords[i * MAX_PATCHES_FILE:(i + 1) * MAX_PATCHES_FILE]
        t0 = time.time()
        patches = [None] * len(chunk)
        with ThreadPoolExecutor(max_workers=_args.workers) as ex:
            futs = {ex.submit(fetch_patch, stack, c[0], c[1]): k for k, c in enumerate(chunk)}
            for n, fut in enumerate(as_completed(futs), 1):
                try:
                    patches[futs[fut]] = fut.result()
                except Exception as exc:
                    print(f'\n    ❌ patch {chunk[futs[fut]]} falhou: {exc}')
                print(f'\r    shard {i + 1}/{n_shards}: {n}/{len(chunk)} patches '
                      f'({time.time() - t0:.0f}s)', end='', flush=True)
        print()
        patches = [p for p in patches if p is not None]
        if not patches:
            print(f'    ⚠️  shard {i + 1} sem patches válidos, não gravado.')
            continue
        tmp = path.with_suffix('.part')
        with tf.io.TFRecordWriter(str(tmp), options=opts) as w:
            for p in patches:
                w.write(to_example(tf, p))
        tmp.rename(path)
        n_fv = sum(int(p['label'].sum() > 0) for p in patches)
        print(f'    salvo → {path.name}  ({len(patches)} patches, {n_fv} com FV)')


def load_fc(asset_id):
    """FeatureCollection do asset; vazia se o asset ainda não existe."""
    try:
        ee.data.getAsset(asset_id)
        return ee.FeatureCollection(asset_id)
    except Exception:
        print(f'  ⚠️  asset não encontrado, usando coleção vazia: {asset_id}')
        return ee.FeatureCollection([])


fcs        = {tipo: load_fc(a) for tipo, a in ASSETS_REVISAO.items()}
fc_confirm = fcs['confirm']
fc_false   = fcs['false']

print(f'\nAno: {YEAR}  |  LABEL_OUTSIDE={LABEL_OUTSIDE}')
print('contando imagens de predict do ano...', flush=True)
n_pred = pred_collection(YEAR).size().getInfo()
print(f'imagens de predict ({YEAR}): {n_pred}')
if n_pred == 0:
    print('⚠️  Nenhum predict encontrado — label ficará só com zeros.'
          ' Confira ASSET_PRED_RURAL (deve ser igual ao do JS).')

stack   = build_stack(YEAR, fc_confirm, fc_false)
out_dir = Path(_args.out_dir).expanduser()

for tipo in _args.tipos:
    # O ano já está no nome do asset; não filtra por 'year'.
    print(f'\n{"=" * 60}\n{tipo}: lendo polígonos ({ASSETS_REVISAO[tipo]})...', flush=True)
    feats = fcs[tipo].getInfo()['features']
    print(f'  {len(feats)} polígonos')
    if not feats:
        continue

    pts, estratos = pontos_por_poligono(feats, tipo, seed=YEAR)
    shards = max(1, math.ceil(len(pts) / MAX_PATCHES_FILE))
    resumo_estratos(tipo, estratos, len(pts))
    print(f'  {len(pts)} pontos  →  {shards} shard(s)')
    if _args.dry_run:
        continue

    if LOCAL:
        out_dir.mkdir(parents=True, exist_ok=True)
        idx = out_dir / f'pontos_{tipo}_{YEAR}_{VERSAO_REVISAO}.csv'
        with open(idx, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=['shard', 'sid', 'classe', 'area_ha', 'n', 'lon', 'lat'])
            w.writeheader()
            for k, p in enumerate(pts):
                w.writerow({'shard': k // MAX_PATCHES_FILE, **p})
        print(f'  índice de pontos → {idx}')
        download_local(stack, [(p['lon'], p['lat']) for p in pts], tipo, out_dir)
        continue

    kernel  = ee.Kernel.rectangle(PATCH_SIZE // 2, PATCH_SIZE // 2, 'pixels')
    patches = stack.neighborhoodToArray(kernel)
    for i in range(shards):
        fname = f'{TASK_PREFIX}{tipo}_{YEAR}_{VERSAO_REVISAO}_part{i:03d}'
        if fname in _completed:
            print(f'    {fname}: já COMPLETED, pulando.')
            continue
        chunk = pts[i * MAX_PATCHES_FILE:(i + 1) * MAX_PATCHES_FILE]
        shard_pts = ee.FeatureCollection(
            [ee.Feature(ee.Geometry.Point([p['lon'], p['lat']])) for p in chunk])
        try:
            export_shard(patches, shard_pts, fname)
            print(f'    enviado → {fname}')
        except Exception as e:
            print(f'    ❌ {fname}: {e}')

print('\nConcluído.' + ('' if LOCAL else ' Acompanhe as tasks no Earth Engine Code Editor.'))
