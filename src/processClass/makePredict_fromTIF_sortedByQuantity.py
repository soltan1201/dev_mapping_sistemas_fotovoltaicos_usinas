#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
makePredict_fromTIF_sortedByQuantity.py
=======================================
Igual ao makePredict_fromTIF.py, mas as regiões são processadas em ordem
decrescente de quantidade de pontos do asset fotovoltaic_rural por grade.

A ordenação usa o GEE Python API para cruzar:
  - grades : projects/nexgenmap/SAD_MapBiomas/DL/SHP_grades_BR_35pathces_AllBrV3
  - fv     : projects/mapbiomas-arida/fotovoltaic_rural

Opcional: --quantity-cache <arquivo.json> salva/reutiliza o resultado do GEE
para evitar chamadas repetidas.

Uso:
  python makePredict_fromTIF_sortedByQuantity.py \\
      --model-path  models/best_5L_unet_efficientnetb7_20260520_2241.keras \\
      --input-dir   ~/teste_dash/mosaicos_tif \\
      [--stride 200] [--batch-size 8] [--threshold 0.5] \\
      [--years 2022 2024] [--regions 00000000000000000005] \\
      [--quantity-cache quantidade_grades.json] \\
      [--gee-project geo-datasciencesol]
"""

import sys
import json
import re
import logging
import argparse
import numpy as np
from pathlib import Path

import tensorflow as tf
for _gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_gpu, True)
import keras
import keras.ops as kops
tf.keras.mixed_precision.set_global_policy('mixed_float16')
print(f'TensorFlow: {tf.__version__}  |  Keras: {keras.__version__}')

import rasterio
from rasterio.crs import CRS as RasterioCRS

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(it, **kw):
        return it

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from custom_losses import (           # noqa: F401 — garante registro dos objetos
    dice_coef, dice_loss,
    focal_tversky_loss, boundary_loss, focal_tversky_boundary_loss,
)
from segmentation_model_factory import ResizeLike, bce_dice_loss, hybrid_focal_loss

# ==============================================================================
# 1. LOGGING
# ==============================================================================
LOG_FILE = Path('predict_fromtif_sorted.log')
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
# 2. CUSTOM OBJECTS
# ==============================================================================
@keras.utils.register_keras_serializable(package='Custom')
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = kops.reshape(kops.cast(y_true, 'float32'), [-1])
    y_pred_f = kops.reshape(kops.cast(y_pred, 'float32'), [-1])
    intersection = kops.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (kops.sum(y_true_f) + kops.sum(y_pred_f) + smooth)

@keras.utils.register_keras_serializable(package='Custom')
def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

@keras.utils.register_keras_serializable(package='RemoteSensing')
def focal_tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, gamma=1.25, smooth=1e-6):
    y_true   = kops.cast(y_true, 'float32')
    y_pred   = kops.cast(y_pred, 'float32')
    y_true_f = kops.reshape(y_true, [-1])
    y_pred_f = kops.reshape(y_pred, [-1])
    tp = kops.sum(y_true_f * y_pred_f)
    fp = kops.sum((1 - y_true_f) * y_pred_f)
    fn = kops.sum(y_true_f * (1 - y_pred_f))
    tversky_index = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return kops.power((1 - tversky_index), gamma)

@keras.utils.register_keras_serializable(package='RemoteSensing')
def boundary_loss(y_true, y_pred, smooth=1e-6):
    y_true   = kops.cast(y_true, 'float32')
    y_pred   = kops.cast(y_pred, 'float32')
    dilated  =  tf.nn.max_pool2d( y_true, ksize=3, strides=1, padding='SAME')
    eroded   = -tf.nn.max_pool2d(-y_true, ksize=3, strides=1, padding='SAME')
    boundary = dilated - eroded
    p        = kops.clip(y_pred, 1e-7, 1.0 - 1e-7)
    bce      = -(y_true * kops.log(p) + (1 - y_true) * kops.log(1 - p))
    return kops.sum(bce * boundary) / (kops.sum(boundary) + smooth)

@keras.utils.register_keras_serializable(package='RemoteSensing')
def focal_tversky_boundary_loss(y_true, y_pred,
                                 alpha=0.3, beta=0.7, gamma=1.25,
                                 boundary_weight=0.85, smooth=1e-6):
    tversky  = focal_tversky_loss(y_true, y_pred, alpha=alpha, beta=beta,
                                   gamma=gamma, smooth=smooth)
    boundary = boundary_loss(y_true, y_pred, smooth=smooth)
    return tversky + boundary_weight * boundary

CUSTOM_OBJECTS = {
    'dice_coef':                    dice_coef,
    'focal_tversky_boundary_loss':  focal_tversky_boundary_loss,
    'ResizeLike':                   ResizeLike,
}

# ==============================================================================
# 3. CONSTANTES
# ==============================================================================
NORM_FACTOR = 10_000.0
PATCH_SIZE  = 256
N_BANDS     = 5

ASSET_GRADE        = 'projects/nexgenmap/SAD_MapBiomas/DL/SHP_grades_BR_35pathces_AllBrV3'
ASSET_PHOTOVOLTAIC = 'projects/mapbiomas-arida/fotovoltaic_rural'

# ==============================================================================
# 4. GEE — QUANTIDADE DE PONTOS POR GRADE
# ==============================================================================

def fetch_region_quantities(gee_project: str, cache_path: Path | None) -> dict:
    """
    Retorna dict {system_index: quantidade} com o número de pontos de
    fotovoltaic_rural dentro de cada grade.

    Usa cache_path como JSON para evitar chamadas repetidas ao GEE.
    """
    if cache_path and cache_path.exists():
        log.info(f'Carregando quantidades do cache: {cache_path}')
        return json.loads(cache_path.read_text(encoding='utf-8'))

    log.info('Conectando ao GEE para obter quantidade por grade…')
    try:
        import ee
    except ImportError:
        raise ImportError(
            'earthengine-api não instalado. '
            'Instale com: pip install earthengine-api'
        )

    ee.Initialize(project=gee_project)

    colection_fv = ee.FeatureCollection(ASSET_PHOTOVOLTAIC)
    grid_shp     = ee.FeatureCollection(ASSET_GRADE)

    # filtra apenas grades que têm pelo menos 1 ponto de fotovoltaica
    grades = grid_shp.filterBounds(colection_fv)

    # adiciona propriedade 'quantidade' = número de pontos de fv dentro de cada grade
    def add_quantity(feat):
        quant = colection_fv.filterBounds(feat.geometry()).size()
        return feat.set('quantidade', quant)

    grades_with_qty = grades.map(add_quantity)

    # traz apenas system:index e quantidade — limita a 5000 features
    features = grades_with_qty.select(['quantidade']).getInfo()['features']

    quantities: dict = {}
    for feat in features:
        idx = feat['id']  # system:index é exposto como 'id' no getInfo()
        qty = feat['properties'].get('quantidade', 0)
        quantities[idx] = qty

    log.info(f'Grades com fotovoltaica: {len(quantities)}')

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(quantities, indent=2, ensure_ascii=False), encoding='utf-8'
        )
        log.info(f'Cache salvo em: {cache_path}')

    return quantities

# ==============================================================================
# 5. MODELO
# ==============================================================================

def _extract_model_key(model_path: Path) -> str:
    stem = model_path.stem
    key  = re.sub(r'_\d{8}_\d{4}$', '', stem)
    key  = re.sub(r'^best_(\d+[A-Za-z]+_)?', '', key)
    return key or stem


def _update_index(input_dir: Path, model_key: str, output_dir: Path) -> Path:
    input_dir  = input_dir.expanduser().resolve()
    output_dir = output_dir.resolve()
    index_path = input_dir.parent / 'predict_index.json'

    index: dict = {}
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding='utf-8'))
        except Exception:
            pass

    index['mosaicos'] = str(input_dir)
    index.setdefault('saidas', {})[model_key] = str(output_dir)

    index_path.write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    log.info(f'Índice atualizado: {index_path}')
    return index_path


def resolve_model_path(model_path: Path) -> Path:
    if model_path.exists():
        return model_path.resolve()
    alt = _HERE.parent / model_path
    if alt.exists():
        return alt.resolve()
    alt2 = _HERE.parent / 'models' / model_path.name
    if alt2.exists():
        return alt2.resolve()
    available = '\n  '.join(str(p) for p in sorted((_HERE.parent / 'models').glob('*.keras')))
    raise FileNotFoundError(
        f"Modelo não encontrado: {model_path}\n"
        f"Disponíveis em {_HERE.parent / 'models'}:\n  {available or '(nenhum)'}"
    )


def load_model(model_path: str):
    resolved = resolve_model_path(Path(model_path))
    log.info(f'Carregando modelo: {resolved}')
    model = tf.keras.models.load_model(str(resolved), custom_objects=CUSTOM_OBJECTS)
    log.info(f'Modelo carregado  |  input={model.input_shape}  output={model.output_shape}')
    return model

# ==============================================================================
# 6. PATCHES E INFERÊNCIA
# ==============================================================================

def patch_origins(size: int, patch: int, stride: int) -> list:
    """Origens dos patches em um eixo; o último patch sempre cobre a borda."""
    if size <= patch:
        return [0]
    origins = list(range(0, size - patch, stride))
    if not origins or origins[-1] + patch < size:
        origins.append(size - patch)
    return origins


def predict_tif(model, tif_path: Path, output_path: Path,
                stride: int, batch_size: int, threshold: float,
                overwrite: bool):
    if output_path.exists() and not overwrite:
        log.info(f'  já existe, pulando: {output_path.name}')
        return

    log.info(f'  lendo: {tif_path.name}')
    with rasterio.open(tif_path) as src:
        img       = src.read()          # (C, H, W) int16
        transform = src.transform
        crs       = src.crs

    img = np.transpose(img.astype(np.float32), (1, 2, 0)) / NORM_FACTOR
    H, W, C = img.shape

    if C != N_BANDS:
        log.warning(f'  bandas encontradas={C}, esperado={N_BANDS}. Usando as {N_BANDS} primeiras.')
        img = img[:, :, :N_BANDS]

    rows = patch_origins(H, PATCH_SIZE, stride)
    cols = patch_origins(W, PATCH_SIZE, stride)
    total = len(rows) * len(cols)
    log.info(f'  imagem {H}×{W}px | patches {len(rows)}×{len(cols)}={total} (stride={stride}px)')

    pred_acc   = np.zeros((H, W), dtype=np.float64)
    weight_acc = np.zeros((H, W), dtype=np.float64)

    batch_patches: list = []
    batch_coords:  list = []

    def flush():
        if not batch_patches:
            return
        arr   = np.stack(batch_patches)
        preds = model.predict(arr, verbose=0)
        for (r0, c0), pred in zip(batch_coords, preds[:, :, :, 0]):
            r1, c1 = min(r0 + PATCH_SIZE, H), min(c0 + PATCH_SIZE, W)
            ph, pw = r1 - r0, c1 - c0
            pred_acc  [r0:r1, c0:c1] += pred[:ph, :pw].astype(np.float64)
            weight_acc[r0:r1, c0:c1] += 1.0
        batch_patches.clear()
        batch_coords.clear()

    patch_iter = tqdm(
        [(r, c) for r in rows for c in cols],
        desc=tif_path.stem, unit='patch', disable=not HAS_TQDM,
    )
    for r0, c0 in patch_iter:
        r1, c1 = min(r0 + PATCH_SIZE, H), min(c0 + PATCH_SIZE, W)
        patch = np.zeros((PATCH_SIZE, PATCH_SIZE, N_BANDS), dtype=np.float32)
        patch[:r1 - r0, :c1 - c0] = img[r0:r1, c0:c1]
        batch_patches.append(patch)
        batch_coords.append((r0, c0))
        if len(batch_patches) == batch_size:
            flush()

    flush()

    pred_map = (pred_acc / np.where(weight_acc == 0, 1.0, weight_acc)).astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if threshold > 0:
        data  = (pred_map > threshold).astype(np.uint8)
        dtype = np.uint8
        log.info(f'  threshold={threshold}  positivos={int(data.sum()):,}px')
    else:
        data  = pred_map
        dtype = np.float32

    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=H, width=W,
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        compress='lzw',
    ) as dst:
        dst.write(data, 1)

    log.info(f'  salvo → {output_path.name}  ({H}×{W}px)')

# ==============================================================================
# 7. PIPELINE PRINCIPAL
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Inferência end-to-end a partir de GeoTIFFs — regiões ordenadas '
                    'por quantidade de pontos fotovoltaicos no GEE')
    parser.add_argument('--model-path', type=Path, required=True,
                        help='Caminho para o .keras do modelo treinado')
    parser.add_argument('--input-dir',  type=Path, required=True,
                        help='Diretório com os TIFs (<region>_<year>.tif)')
    parser.add_argument('--stride',     type=int,   default=200,
                        help='Stride em pixels entre patches (padrão: 200)')
    parser.add_argument('--batch-size', type=int,   default=8,
                        help='Patches por batch de inferência (padrão: 8)')
    parser.add_argument('--threshold',  type=float, default=0.5,
                        help='Threshold de binarização; 0 = float32 probabilidade (padrão: 0.5)')
    parser.add_argument('--years',      type=int, nargs='+', default=None,
                        help='Anos a filtrar (padrão: todos)')
    parser.add_argument('--regions',    type=str, nargs='+', default=None,
                        help='IDs de região a filtrar (padrão: todos)')
    parser.add_argument('--overwrite',  action='store_true',
                        help='Reprocessa mesmo que o TIF de saída já exista')
    parser.add_argument('--quantity-cache', type=Path, default=None,
                        help='JSON para salvar/carregar quantidade por grade (evita chamada GEE)')
    parser.add_argument('--gee-project', type=str, default='geo-datasciencesol',
                        help='Projeto GEE para inicialização (padrão: geo-datasciencesol)')
    args = parser.parse_args()

    model_key  = _extract_model_key(args.model_path)
    input_dir  = args.input_dir.expanduser().resolve()
    output_dir = input_dir.parent / f'{input_dir.name}_{model_key}'

    gpus = tf.config.list_physical_devices('GPU')
    log.info(f'GPU(s): {[g.name for g in gpus] or "nenhuma — CPU"}')
    log.info('=' * 60)
    log.info(f'Modelo      : {args.model_path}')
    log.info(f'Chave modelo: {model_key}')
    log.info(f'Entrada     : {input_dir}')
    log.info(f'Saída       : {output_dir}')
    log.info(f'Stride      : {args.stride}px')
    log.info(f'Batch size  : {args.batch_size}')
    log.info(f'Threshold   : {args.threshold}')
    log.info(f'Anos        : {args.years or "todos"}')
    log.info(f'Regiões     : {args.regions or "todas"}')
    log.info(f'GEE project : {args.gee_project}')
    log.info(f'Cache qtd   : {args.quantity_cache or "não usado"}')
    log.info('=' * 60)

    # ------------------------------------------------------------------
    # Obtém quantidades por grade (GEE ou cache)
    # ------------------------------------------------------------------
    quantities = fetch_region_quantities(args.gee_project, args.quantity_cache)

    model = load_model(str(args.model_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    tif_files = sorted(input_dir.glob('*.tif'))
    if not tif_files:
        log.error(f'Nenhum .tif encontrado em {input_dir}')
        return

    # Filtra por ano / região
    filtered: list = []
    for tif_path in tif_files:
        parts = tif_path.stem.rsplit('_', 1)
        if len(parts) != 2 or not parts[1].isdigit():
            log.warning(f'Nome não reconhecido, pulando: {tif_path.name}')
            continue
        region_id, year = parts[0], int(parts[1])
        if args.years and year not in args.years:
            continue
        if args.regions and region_id not in args.regions:
            continue
        filtered.append((tif_path, region_id, int(year)))

    # ------------------------------------------------------------------
    # Ordena por quantidade decrescente (mais pontos fotovoltaicos primeiro)
    # Regiões sem quantidade no GEE ficam no final (quantidade = 0)
    # ------------------------------------------------------------------
    filtered.sort(key=lambda x: quantities.get(x[1], 0), reverse=True)

    region_order: dict = {}
    for _, rid, _ in filtered:
        if rid not in region_order:
            region_order[rid] = len(region_order) + 1
    total_regions = len(region_order)

    log.info(f'TIFs a processar: {len(filtered)} ({total_regions} regiões) — ordenados por quantidade')

    # Log da ordem das regiões
    seen_order: set = set()
    for _, rid, _ in filtered:
        if rid not in seen_order:
            seen_order.add(rid)
            qty = quantities.get(rid, 0)
            log.info(f'  [{region_order[rid]:>3}/{total_regions}] {rid}  qtd={qty}')

    log.info('=' * 60)

    processed = 0
    for tif_path, region_id, year in filtered:
        region_idx = region_order[region_id]
        qty        = quantities.get(region_id, 0)

        log.info(f'\n{"="*60}')
        log.info(f'Região: {region_id} ({region_idx}/{total_regions})  |  Ano: {year}  |  qtd_fv={qty}')

        try:
            predict_tif(
                model       = model,
                tif_path    = tif_path,
                output_path = output_dir / f'pred_{region_id}_{year}.tif',
                stride      = args.stride,
                batch_size  = args.batch_size,
                threshold   = args.threshold,
                overwrite   = args.overwrite,
            )
            processed += 1
        except Exception as exc:
            log.error(f'  Erro em {tif_path.name}: {exc}', exc_info=True)

    log.info(f'\nConcluído. {processed} TIF(s) processado(s). Log: {LOG_FILE.resolve()}')

    _update_index(input_dir, model_key, output_dir)


if __name__ == '__main__':
    main()
