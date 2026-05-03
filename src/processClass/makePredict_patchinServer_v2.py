#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inferência local — Fotovoltaica (Planet NICFI)
================================================
Roda predição sobre patches 256×256×8 gerados por:
  • download_dataset_predict_fotovoltaica.py  →  --input-format npy
  • convert_npy_to_tfrecord_fotovoltaica.py   →  --input-format tfrecord

Uso:
  python makePredict_patchinServer_v2.py \\
      --model-path  /modelos/best_unet_resnet101.keras \\
      --input-dir   /dados/dataset_fotovoltaica_npy \\
      --input-format npy \\
      --output-dir  /dados/predict_fotovoltaica \\
      --threshold   0.5 \\
      --batch-size  8

Saída por patch:
  <output-dir>/<region_id>/<year>/patch_r<row>_c<col>_pred.npy   →  (256,256) float32 [0,1]
  <output-dir>/<region_id>/<year>/patch_r<row>_c<col>_pred.json  →  metadados + params
"""

import sys
import os
import json
import logging
import argparse
import numpy as np
from pathlib import Path

import tensorflow as tf
import keras
import keras.ops as kops

print(f'TensorFlow: {tf.__version__}  |  Keras: {keras.__version__}')

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(it, **kw):
        return it

# Adiciona o diretório deste script ao path para importar a factory
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from segmentation_model_factory import (
    ResizeLike,
    dice_coef,
    dice_loss,
    bce_dice_loss,
    hybrid_focal_loss,
)

# ==============================================================================
# 1. LOGGING
# ==============================================================================

LOG_FILE = Path('predict_fotovoltaica.log')
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
# 2. CUSTOM OBJECTS — definidos no colab (cell 23) mas não na factory
#    Precisam ser registrados ANTES de tf.keras.models.load_model()
# ==============================================================================

@keras.utils.register_keras_serializable(package='RemoteSensing')
def focal_tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, gamma=1.25, smooth=1e-6):
    """Focal Tversky Loss — alpha=0.3, beta=0.7 foca em Recall (reduz FN)."""
    y_true   = kops.cast(y_true, 'float32')
    y_pred   = kops.cast(y_pred, 'float32')
    y_true_f = kops.reshape(y_true, [-1])
    y_pred_f = kops.reshape(y_pred, [-1])
    tp = kops.sum(y_true_f * y_pred_f)
    fp = kops.sum((1 - y_true_f) * y_pred_f)
    fn = kops.sum(y_true_f * (1 - y_pred_f))
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return kops.power((1 - tversky), gamma)


@keras.utils.register_keras_serializable(package='RemoteSensing')
def boundary_loss(y_true, y_pred, smooth=1e-6):
    """BCE concentrado nos pixels de borda do GT (morfologia 3×3 diferenciável)."""
    y_true = kops.cast(y_true, 'float32')
    y_pred = kops.cast(y_pred, 'float32')
    dilated  =  tf.nn.max_pool2d( y_true, ksize=3, strides=1, padding='SAME')
    eroded   = -tf.nn.max_pool2d(-y_true, ksize=3, strides=1, padding='SAME')
    boundary = dilated - eroded
    p   = kops.clip(y_pred, 1e-7, 1.0 - 1e-7)
    bce = -(y_true * kops.log(p) + (1 - y_true) * kops.log(1 - p))
    return kops.sum(bce * boundary) / (kops.sum(boundary) + smooth)


@keras.utils.register_keras_serializable(package='RemoteSensing')
def focal_tversky_boundary_loss(y_true, y_pred,
                                 alpha=0.3, beta=0.7, gamma=1.25,
                                 boundary_weight=0.85, smooth=1e-6):
    """Focal Tversky + Boundary Loss (loss principal usada no treinamento)."""
    tversky  = focal_tversky_loss(y_true, y_pred,
                                   alpha=alpha, beta=beta, gamma=gamma, smooth=smooth)
    boundary = boundary_loss(y_true, y_pred, smooth=smooth)
    return tversky + boundary_weight * boundary


# Mapa completo de custom objects para load_model
CUSTOM_OBJECTS = {
    # Da factory
    'ResizeLike':                  ResizeLike,
    'dice_coef':                   dice_coef,
    'dice_loss':                   dice_loss,
    'bce_dice_loss':               bce_dice_loss,
    'hybrid_focal_loss':           hybrid_focal_loss,
    # Do colab (cell 23)
    'focal_tversky_loss':          focal_tversky_loss,
    'boundary_loss':               boundary_loss,
    'focal_tversky_boundary_loss': focal_tversky_boundary_loss,
    # Compatibilidade com modelos mais antigos (v1)
    'jaccard_index': lambda y_true, y_pred: (
        lambda pred: tf.reduce_sum(y_true * pred) /
                     (tf.reduce_sum(y_true) + tf.reduce_sum(pred)
                      - tf.reduce_sum(y_true * pred) + tf.keras.backend.epsilon())
    )(tf.cast(y_pred > 0.5, tf.float32)),
}

# ==============================================================================
# 3. CONFIGURAÇÕES
# ==============================================================================

NORM_FACTOR  = 10_000.0
PATCH_SIZE   = 256
N_BANDS      = 8
FEATURE_BANDS = ['blue', 'green', 'red', 'nir', 'pvi', 'iia', 'ri', 'evi']

# Feature spec para TFRecords gerados por convert_npy_to_tfrecord_fotovoltaica.py
TFRECORD_FEATURE_SPEC = {
    'patch/data':     tf.io.FixedLenFeature([], tf.string),
    'patch/shape':    tf.io.FixedLenFeature([3], tf.int64),
    'meta/region_id': tf.io.FixedLenFeature([], tf.string),
    'meta/year':      tf.io.FixedLenFeature([1], tf.int64),
    'meta/row':       tf.io.FixedLenFeature([1], tf.int64),
    'meta/col':       tf.io.FixedLenFeature([1], tf.int64),
    'meta/transform': tf.io.FixedLenFeature([6], tf.float32),
    'meta/crs':       tf.io.FixedLenFeature([], tf.string),
}

# ==============================================================================
# 4. FUNÇÕES — MODELO
# ==============================================================================

def load_model(model_path: str):
    log.info(f'Carregando modelo: {model_path}')
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=CUSTOM_OBJECTS)
        log.info(f'Modelo carregado  |  input={model.input_shape}  output={model.output_shape}')
        return model
    except Exception as exc:
        log.error(f'Falha ao carregar modelo: {exc}')
        raise


def setup_device():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        log.info(f'GPU(s) disponível(is): {[g.name for g in gpus]}')
    else:
        log.warning('Nenhuma GPU encontrada — executando em CPU (mais lento).')

# ==============================================================================
# 5. FUNÇÕES — SALVAMENTO
# ==============================================================================

def save_prediction(pred_hw: np.ndarray, meta: dict, out_dir: Path,
                    threshold: float, model_path: str):
    """Salva mapa de probabilidade (.npy float32) e metadados (.json)."""
    fname     = f"patch_r{meta['row']:04d}_c{meta['col']:04d}_pred"
    npy_path  = out_dir / f'{fname}.npy'
    json_path = out_dir / f'{fname}.json'

    np.save(npy_path, pred_hw.astype(np.float32))

    json_path.write_text(json.dumps({
        **meta,
        'threshold':  threshold,
        'model_path': str(model_path),
        'bands':      FEATURE_BANDS,
        'dtype':      'float32',
        'shape':      list(pred_hw.shape),
    }, indent=2, ensure_ascii=False))

# ==============================================================================
# 6. MODO NPY
# ==============================================================================

def iter_npy_batches(input_dir: Path, output_dir: Path,
                     batch_size: int, filter_years, filter_regions):
    """Gera batches (arrays_f32, metas, out_dirs) varrendo a estrutura de pastas."""
    batch_arrays, batch_metas, batch_outdirs = [], [], []

    for region_dir in sorted(input_dir.iterdir()):
        if not region_dir.is_dir():
            continue
        if filter_regions and region_dir.name not in filter_regions:
            continue

        for year_dir in sorted(region_dir.iterdir()):
            if not year_dir.is_dir():
                continue
            if filter_years and int(year_dir.name) not in filter_years:
                continue

            out_dir = output_dir / region_dir.name / year_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)

            for npy_path in sorted(year_dir.glob('patch_*.npy')):
                json_path = npy_path.with_suffix('.json')
                pred_path = out_dir / (npy_path.stem + '_pred.npy')
                if pred_path.exists():
                    continue  # resume

                try:
                    arr  = np.load(npy_path).astype(np.float32) / NORM_FACTOR
                    meta = json.loads(json_path.read_text(encoding='utf-8'))
                except Exception as exc:
                    log.error(f'Erro ao ler {npy_path.name}: {exc}')
                    continue

                batch_arrays.append(arr)
                batch_metas.append(meta)
                batch_outdirs.append(out_dir)

                if len(batch_arrays) == batch_size:
                    yield np.stack(batch_arrays), batch_metas, batch_outdirs
                    batch_arrays, batch_metas, batch_outdirs = [], [], []

    if batch_arrays:
        yield np.stack(batch_arrays), batch_metas, batch_outdirs


def predict_npy(model, input_dir: Path, output_dir: Path,
                batch_size: int, threshold: float,
                filter_years, filter_regions, model_path: str):
    log.info(f'[NPY] Fonte: {input_dir}')

    batches   = list(iter_npy_batches(input_dir, output_dir,
                                      batch_size, filter_years, filter_regions))
    n_patches = sum(len(m) for _, m, _ in batches)
    log.info(f'Patches a processar: {n_patches}  |  batches: {len(batches)}')

    total_saved = 0
    for batch_arr, batch_metas, batch_outdirs in tqdm(batches, unit='batch',
                                                       disable=not HAS_TQDM):
        preds = model.predict(batch_arr, verbose=0)   # (B, 256, 256, 1)
        for pred, meta, out_dir in zip(preds[:, :, :, 0], batch_metas, batch_outdirs):
            save_prediction(pred, meta, out_dir, threshold, model_path)
            total_saved += 1

    log.info(f'[NPY] Predições salvas: {total_saved}')

# ==============================================================================
# 7. MODO TFRECORD
# ==============================================================================

def _decode_tfrecord(proto):
    """Decodifica um Example e retorna (patch_f32 [H,W,C], meta_dict)."""
    parsed = tf.io.parse_single_example(proto, TFRECORD_FEATURE_SPEC)
    patch  = tf.io.decode_raw(parsed['patch/data'], tf.float32)
    patch  = tf.reshape(patch, parsed['patch/shape'])   # (256, 256, 8) float32 [0,1]
    return patch, parsed


def predict_tfrecord(model, input_dir: Path, output_dir: Path,
                     batch_size: int, threshold: float,
                     filter_years, filter_regions, model_path: str):
    log.info(f'[TFRecord] Fonte: {input_dir}')

    total_saved = 0

    for region_dir in sorted(input_dir.iterdir()):
        if not region_dir.is_dir():
            continue
        if filter_regions and region_dir.name not in filter_regions:
            continue

        for year_dir in sorted(region_dir.iterdir()):
            if not year_dir.is_dir():
                continue
            if filter_years and int(year_dir.name) not in filter_years:
                continue

            tfrecord_files = sorted(str(p) for p in year_dir.glob('*.tfrecord'))
            if not tfrecord_files:
                continue

            out_dir = output_dir / region_dir.name / year_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)

            log.info(f'  {region_dir.name}/{year_dir.name}  — {len(tfrecord_files)} shard(s)')

            ds = (tf.data.TFRecordDataset(tfrecord_files,
                                          num_parallel_reads=tf.data.AUTOTUNE)
                  .map(_decode_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
                  .batch(batch_size)
                  .prefetch(tf.data.AUTOTUNE))

            for batch_patches, batch_meta in ds:
                preds = model.predict(batch_patches, verbose=0)   # (B, 256, 256, 1)
                b     = preds.shape[0]

                for i in range(b):
                    row = int(batch_meta['meta/row'][i].numpy()[0])
                    col = int(batch_meta['meta/col'][i].numpy()[0])

                    pred_path = out_dir / f'patch_r{row:04d}_c{col:04d}_pred.npy'
                    if pred_path.exists():
                        continue  # resume

                    meta = {
                        'region_id': batch_meta['meta/region_id'][i].numpy().decode(),
                        'year':      int(batch_meta['meta/year'][i].numpy()[0]),
                        'row':       row,
                        'col':       col,
                        'transform': batch_meta['meta/transform'][i].numpy().tolist(),
                        'crs':       batch_meta['meta/crs'][i].numpy().decode(),
                    }
                    save_prediction(preds[i, :, :, 0], meta, out_dir,
                                    threshold, model_path)
                    total_saved += 1

    log.info(f'[TFRecord] Predições salvas: {total_saved}')

# ==============================================================================
# 8. PIPELINE PRINCIPAL
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Inferência UNet fotovoltaica sobre patches locais')
    parser.add_argument('--model-path',   type=Path, required=True,
                        help='Caminho para o arquivo .keras do modelo treinado')
    parser.add_argument('--input-dir',    type=Path, required=True,
                        help='Diretório raiz dos patches (NPY ou TFRecord)')
    parser.add_argument('--input-format', choices=['npy', 'tfrecord'], default='npy',
                        help='Formato dos patches de entrada (padrão: npy)')
    parser.add_argument('--output-dir',   type=Path, required=True,
                        help='Diretório de saída das predições')
    parser.add_argument('--threshold',    type=float, default=0.5,
                        help='Limiar de binarização salvo nos metadados (padrão: 0.5)')
    parser.add_argument('--batch-size',   type=int, default=8,
                        help='Patches por batch de inferência (padrão: 8)')
    parser.add_argument('--years',        type=int, nargs='+', default=None,
                        help='Anos a processar (padrão: todos)')
    parser.add_argument('--regions',      type=str, nargs='+', default=None,
                        help='IDs de região a processar (padrão: todos)')
    args = parser.parse_args()

    log.info('=' * 60)
    log.info(f'Modelo      : {args.model_path}')
    log.info(f'Entrada     : {args.input_dir}  [{args.input_format}]')
    log.info(f'Saída       : {args.output_dir}')
    log.info(f'Threshold   : {args.threshold}')
    log.info(f'Batch size  : {args.batch_size}')
    log.info(f'Anos        : {args.years or "todos"}')
    log.info(f'Regiões     : {args.regions or "todas"}')
    log.info('=' * 60)

    setup_device()

    model = load_model(str(args.model_path))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_format == 'npy':
        predict_npy(
            model        = model,
            input_dir    = args.input_dir,
            output_dir   = args.output_dir,
            batch_size   = args.batch_size,
            threshold    = args.threshold,
            filter_years = args.years,
            filter_regions = args.regions,
            model_path   = str(args.model_path),
        )
    else:
        predict_tfrecord(
            model        = model,
            input_dir    = args.input_dir,
            output_dir   = args.output_dir,
            batch_size   = args.batch_size,
            threshold    = args.threshold,
            filter_years = args.years,
            filter_regions = args.regions,
            model_path   = str(args.model_path),
        )

    log.info(f'Concluído. Log: {LOG_FILE.resolve()}')


if __name__ == '__main__':
    main()
