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
      --model-path  /modelos/best_unet_resnet50_20260430_0257.keras \\
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
# memory_growth deve ser configurado antes de qualquer operação que inicialize o contexto CUDA
for _gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(_gpu, True)
import keras
import keras.ops as kops
tf.keras.mixed_precision.set_global_policy('mixed_float16')
print(f'TensorFlow: {tf.__version__}  |  Keras: {keras.__version__}')

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(it, **kw):
        return it

# Registra todos os custom objects com os packages corretos (igual ao Colab)
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
pathparent = str(Path(os.getcwd()).parents[0])
print(pathparent)
from custom_losses import (
    build_custom_objects,
    dice_coef, dice_loss,
    focal_tversky_loss, boundary_loss, focal_tversky_boundary_loss,
)
from segmentation_model_factory import ResizeLike, bce_dice_loss, hybrid_focal_loss
# sys.exit()
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
# 2. CUSTOM OBJECTS
# ==============================================================================
## célula no. 24
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
    """
    Focal Tversky Loss — alpha=0.3, beta=0.7 foca em Recall (reduz FN).
    Captura estruturas finas/esparsas que Dice/BCE ignoram.
    """
    y_true = kops.cast(y_true, 'float32')
    y_pred = kops.cast(y_pred, 'float32')
    y_true_f = kops.reshape(y_true, [-1])
    y_pred_f = kops.reshape(y_pred, [-1])
    tp = kops.sum(y_true_f * y_pred_f)
    fp = kops.sum((1 - y_true_f) * y_pred_f)
    fn = kops.sum(y_true_f * (1 - y_pred_f))
    tversky_index = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return kops.power((1 - tversky_index), gamma)


@keras.utils.register_keras_serializable(package='RemoteSensing')
def boundary_loss(y_true, y_pred, smooth=1e-6):
    """
    BCE concentrado nos pixels de borda do GT.
    Borda = dilatação morfológica − erosão (kernel 3×3).
    Força o modelo a ser preciso nos contornos — reduz bordas arredondadas.
    """
    y_true = kops.cast(y_true, 'float32')
    y_pred = kops.cast(y_pred, 'float32')

    # Morfologia diferenciável via max-pool 3×3
    dilated  =  tf.nn.max_pool2d( y_true, ksize=3, strides=1, padding='SAME')
    eroded   = -tf.nn.max_pool2d(-y_true, ksize=3, strides=1, padding='SAME')
    boundary = dilated - eroded   # (B, H, W, 1) — pixels exatamente na borda do GT

    # BCE pixel a pixel estável numericamente
    p   = kops.clip(y_pred, 1e-7, 1.0 - 1e-7)
    bce = -(y_true * kops.log(p) + (1 - y_true) * kops.log(1 - p))

    # Gradiente concentrado nos pixels de borda
    return kops.sum(bce * boundary) / (kops.sum(boundary) + smooth)


@keras.utils.register_keras_serializable(package='RemoteSensing')
def focal_tversky_boundary_loss(y_true, y_pred,
                                 alpha=0.3, beta=0.7, gamma=1.25,
                                 boundary_weight=0.85, smooth=1e-6):
    """
    Focal Tversky  +  Boundary Loss.
      Tversky  (peso 1.0) → captura estruturas finas, penaliza FN
      Boundary (peso 0.5) → nitidez nas bordas, penaliza erros no contorno
    Total ≈ 67 % Tversky + 33 % Boundary.
    """
    tversky  = focal_tversky_loss(y_true, y_pred, alpha=alpha, beta=beta,
                                   gamma=gamma, smooth=smooth)
    boundary = boundary_loss(y_true, y_pred, smooth=smooth)
    return tversky + boundary_weight * boundary


print('Losses e métricas definidas:')
print('  focal_tversky_loss          (alpha=0.3  beta=0.7  gamma=1.25)')
print('  boundary_loss               (kernel 3×3  morfologia diferenciável)')
print('  focal_tversky_boundary_loss (tversky + 0.5 × boundary)')
# CUSTOM_OBJECTS = build_custom_objects()
CUSTOM_OBJECTS = {
    'dice_coef': dice_coef, # métrica
    'focal_tversky_boundary_loss': focal_tversky_boundary_loss, # loss
    'ResizeLike': ResizeLike # Adiciona a camada customizada
}


# ==============================================================================
# 3. CONFIGURAÇÕES
# ==============================================================================

NORM_FACTOR   = 10_000.0
PATCH_SIZE    = 256
N_BANDS       = 5
# FEATURE_BANDS = ['blue', 'green', 'red', 'nir', 'pvi', 'iia', 'ri', 'evi']
FEATURE_BANDS = ['blue', 'green', 'red', 'pvi', 'pvpi']

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

# Feature spec para TFRecords gerados por download_predict_as_tfrecord_fv.py
# (Export.table.toDrive + neighborhoodToArray — rect(128) → 257×257 por banda)
_GEE_PATCH_FLAT = 257 * 257   # 66 049 valores por banda
GEE_TFRECORD_FEATURE_SPEC = {
    **{b: tf.io.FixedLenFeature([_GEE_PATCH_FLAT], tf.float32) for b in FEATURE_BANDS},
    'region_id': tf.io.FixedLenFeature([], tf.string),
    'year':      tf.io.FixedLenFeature([], tf.float32),
    'latitude':  tf.io.FixedLenFeature([], tf.float32),
    'longitude': tf.io.FixedLenFeature([], tf.float32),
}

# ==============================================================================
# 4. FUNÇÕES — MODELO
# ==============================================================================

def resolve_model_path(model_path: Path) -> Path:
    """Resolve o caminho do modelo: tenta CWD, depois src/models/ ao lado do script."""
    if model_path.exists():
        return model_path.resolve()
    alt = _HERE.parent / model_path          # src/<caminho passado>
    if alt.exists():
        return alt.resolve()
    alt2 = _HERE.parent / 'models' / model_path.name   # src/models/<nome>
    if alt2.exists():
        return alt2.resolve()
    available = '\n  '.join(str(p) for p in sorted((_HERE.parent / 'models').glob('*.keras')))
    raise FileNotFoundError(
        f"Modelo não encontrado: {model_path}\n"
        f"Modelos disponíveis em {_HERE.parent / 'models'}:\n  {available or '(nenhum)'}"
    )


def load_model(model_path: str):
    resolved = resolve_model_path(Path(model_path))
    log.info(f'Carregando modelo: {resolved}')
    try:
        model = tf.keras.models.load_model(str(resolved), custom_objects=CUSTOM_OBJECTS)
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

def predict_npy(model, input_dir: Path, output_dir: Path,
                batch_size: int, threshold: float,
                filter_years, filter_regions, model_path: str):
    log.info(f'[NPY] Fonte: {input_dir}')

    total_saved = 0
    batch_arrays, batch_metas, batch_outdirs = [], [], []

    for region_dir in input_dir.iterdir():
        if not region_dir.is_dir():
            continue
        if filter_regions and region_dir.name not in filter_regions:
            continue

        for year_dir in region_dir.iterdir():
            if not year_dir.is_dir():
                continue
            if not year_dir.name.isdigit():
                continue
            if filter_years and int(year_dir.name) not in filter_years:
                continue

            out_dir = output_dir / region_dir.name / year_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)

            for npy_path in year_dir.glob('patch_*.npy'):
                json_path = npy_path.with_suffix('.json')
                pred_path = out_dir / (npy_path.stem + '_pred.npy')
                if pred_path.exists():
                    continue  # resume

                try:
                    arr  = np.load(npy_path).astype(np.float32) / NORM_FACTOR
                    meta = json.loads(json_path.read_text(encoding='utf-8'))
                    meta.setdefault('region_id', region_dir.name)
                except Exception as exc:
                    log.error(f'Erro ao ler {npy_path.name}: {exc}')
                    continue

                batch_arrays.append(arr)
                batch_metas.append(meta)
                batch_outdirs.append(out_dir)

                if len(batch_arrays) == batch_size:
                    preds = model.predict(np.stack(batch_arrays), verbose=0)
                    for pred, m, odir in zip(preds[:, :, :, 0], batch_metas, batch_outdirs):
                        save_prediction(pred, m, odir, threshold, model_path)
                        total_saved += 1
                    log.info(f'  region: {region_dir.name}  <>  year: {year_dir.name}  <>  salvos: {total_saved}')
                    batch_arrays, batch_metas, batch_outdirs = [], [], []

            # flush patches residuais do year_dir (pasta com menos de batch_size imagens)
            if batch_arrays:
                preds = model.predict(np.stack(batch_arrays), verbose=0)
                for pred, m, odir in zip(preds[:, :, :, 0], batch_metas, batch_outdirs):
                    save_prediction(pred, m, odir, threshold, model_path)
                    total_saved += 1
                log.info(f'  region: {region_dir.name}  <>  year: {year_dir.name}  <>  salvos (residual): {total_saved}')
                batch_arrays, batch_metas, batch_outdirs = [], [], []

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

    # Verificação antecipada: avisa se não há nenhum .tfrecord / .tfrecord.gz no input_dir
    all_tfrecords = [
        p for p in input_dir.rglob('*')
        if p.name.endswith('.tfrecord') or p.name.endswith('.tfrecord.gz')
    ]
    if not all_tfrecords:
        npy_count = sum(1 for _ in input_dir.rglob('patch_*.npy'))
        hint = f' (encontrados {npy_count} arquivos .npy — use --input-format npy)' if npy_count else ''
        log.warning(f'Nenhum arquivo .tfrecord encontrado em {input_dir}{hint}')
        return

    compressed = all_tfrecords[0].name.endswith('.tfrecord.gz')
    log.info(f'[TFRecord] {len(all_tfrecords)} arquivo(s) encontrado(s) | comprimido={compressed}')
    log.info(f'[TFRecord] Exemplo: {all_tfrecords[0].relative_to(input_dir)}')

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

            tfrecord_files = sorted(
                str(p) for p in year_dir.glob('*')
                if p.name.endswith('.tfrecord') or p.name.endswith('.tfrecord.gz')
            )
            if not tfrecord_files:
                continue

            out_dir = output_dir / region_dir.name / year_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)

            log.info(f'  {region_dir.name}/{year_dir.name}  — {len(tfrecord_files)} shard(s)')

            ds = (tf.data.TFRecordDataset(tfrecord_files,
                                          compression_type='GZIP' if compressed else '',
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
# 7b. MODO GEE-TFRECORD (download_predict_as_tfrecord_fv.py)
# ==============================================================================

def save_prediction_gee(pred_hw: np.ndarray, meta: dict, out_dir: Path,
                        threshold: float, model_path: str):
    """Salva predição com nome baseado em lat/lon (sem row/col disponíveis)."""
    lat = meta.get('latitude', 0.0)
    lon = meta.get('longitude', 0.0)
    fname     = f"patch_lat{lat:.5f}_lon{lon:.5f}_pred"
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


def _decode_gee_tfrecord(proto):
    """Decodifica Example do formato GEE export → (patch_f32 [256,256,8], meta_dict)."""
    parsed = tf.io.parse_single_example(proto, GEE_TFRECORD_FEATURE_SPEC)

    # Bandas já são float32; empilha (257×257) e center-crop para 256×256
    bands = [tf.reshape(parsed[b], [257, 257]) for b in FEATURE_BANDS]
    patch = tf.stack(bands, axis=-1)[:256, :256, :]   # (256, 256, 8)
    patch = patch / NORM_FACTOR

    return patch, parsed


def predict_gee_tfrecord(model, input_dir: Path, output_dir: Path,
                          batch_size: int, threshold: float,
                          filter_years, filter_regions, model_path: str):
    """Inferência sobre TFRecords no formato do download_predict_as_tfrecord_fv.py."""
    import re
    from collections import defaultdict

    log.info(f'[GEE-TFRecord] Fonte: {input_dir}')

    all_files = sorted(
        p for p in input_dir.rglob('*.tfrecord*')
        if p.name.endswith('.tfrecord') or p.name.endswith('.tfrecord.gz')
    )
    if not all_files:
        log.warning('Nenhum arquivo .tfrecord / .tfrecord.gz encontrado.')
        return

    compression = 'GZIP' if all_files[0].name.endswith('.gz') else ''
    log.info(f'Arquivos encontrados: {len(all_files)}  (compressão: {compression or "nenhuma"})')

    # Agrupa shards por (region_id, year) via nome: predict_fv_{region}_{year}_part{N}
    pattern = re.compile(r'predict_fv_([0-9a-f]+)_(\d{4})_part\d+')
    groups: dict[tuple[str, int], list[str]] = defaultdict(list)
    for p in all_files:
        m = pattern.search(p.name)
        if not m:
            log.warning(f'Nome não reconhecido, pulando: {p.name}')
            continue
        region_id, year = m.group(1), int(m.group(2))
        if filter_regions and region_id not in filter_regions:
            continue
        if filter_years and year not in filter_years:
            continue
        groups[(region_id, year)].append(str(p))

    log.info(f'Grupos (região × ano) a processar: {len(groups)}')

    total_saved = 0
    for (region_id, year), files in sorted(groups.items()):
        out_dir = output_dir / region_id / str(year)
        out_dir.mkdir(parents=True, exist_ok=True)

        log.info(f'  {region_id}/{year}  — {len(files)} shard(s)')

        ds = (tf.data.TFRecordDataset(files,
                                       compression_type=compression,
                                       num_parallel_reads=min(4, len(files)))
              .map(_decode_gee_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
              .batch(batch_size)
              .prefetch(2))

        for batch_patches, batch_meta in ds:
            # model(...) direto evita o overhead por chamada do model.predict()
            preds = model(batch_patches, training=False).numpy()  # (B, 256, 256, 1)
            b     = preds.shape[0]

            for i in range(b):
                lat = float(batch_meta['latitude'][i].numpy())
                lon = float(batch_meta['longitude'][i].numpy())

                pred_path = out_dir / f'patch_lat{lat:.5f}_lon{lon:.5f}_pred.npy'
                if pred_path.exists():
                    continue  # resume

                meta = {'region_id': region_id, 'year': year,
                        'latitude': lat, 'longitude': lon}
                save_prediction_gee(preds[i, :, :, 0], meta, out_dir,
                                    threshold, model_path)
                total_saved += 1

        log.info(f'    → salvos acumulado: {total_saved}')

    log.info(f'[GEE-TFRecord] Predições salvas: {total_saved}')

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
    parser.add_argument('--input-format',
                        choices=['npy', 'tfrecord', 'gee-tfrecord'], default='npy',
                        help='Formato dos patches: npy | tfrecord | gee-tfrecord (padrão: npy)')
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
    path_model = os.path.join(pathparent, str(args.model_path))
    model = load_model(path_model)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    common = dict(
        model          = model,
        input_dir      = args.input_dir,
        output_dir     = args.output_dir,
        batch_size     = args.batch_size,
        threshold      = args.threshold,
        filter_years   = args.years,
        filter_regions = args.regions,
        model_path     = path_model, # str(args.model_path)
    )

    if args.input_format == 'npy':
        predict_npy(**common)
    elif args.input_format == 'tfrecord':
        predict_tfrecord(**common)
    else:
        predict_gee_tfrecord(**common)

    log.info(f'Concluído. Log: {LOG_FILE.resolve()}')


if __name__ == '__main__':
    main()
