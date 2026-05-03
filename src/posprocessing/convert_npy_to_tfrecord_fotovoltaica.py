#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conversão NPY → TFRecord — Fotovoltaica (Planet NICFI)
=======================================================
Lê os patches .npy gerados por download_dataset_predict_fotovoltaica.py
e os converte para arquivos .tfrecord otimizados para tf.data.

Estrutura de entrada (gerada pelo script de download):
  <NPY_DIR>/<region_id>/<year>/patch_r<row>_c<col>.npy   →  (256, 256, 8) int16
  <NPY_DIR>/<region_id>/<year>/patch_r<row>_c<col>.json  →  metadados

Estrutura de saída:
  <TFRECORD_DIR>/<region_id>/<year>/<region_id>_<year>_part_NNNN.tfrecord

Cada tf.train.Example contém:
  patch/data      bytes    — float32 serializado, shape (256, 256, 8), valores [0, 1]
  patch/shape     int64[3] — [256, 256, 8]
  meta/region_id  bytes    — region_id em UTF-8
  meta/year       int64[1]
  meta/row        int64[1]
  meta/col        int64[1]
  meta/transform  float[6] — [scale_x, shear_x, origin_x, shear_y, -scale_y, origin_y]
  meta/crs        bytes    — string CRS em UTF-8

Decodificação no pipeline de inferência:
  dataset = tf.data.TFRecordDataset(tfrecord_files)
  dataset = dataset.map(decode_example)
"""

import json
import logging
import numpy as np
from pathlib import Path

try:
    import tensorflow as tf
except ImportError:
    raise ImportError("TensorFlow não encontrado. Instale com: pip install tensorflow")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(it, **kw):
        return it

# ==============================================================================
# 1. LOGGING
# ==============================================================================

LOG_FILE = Path('npy_to_tfrecord.log')
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

NPY_DIR      = Path('/dados/dataset_fotovoltaica_npy')
TFRECORD_DIR = Path('/dados/dataset_fotovoltaica_tfrecord')

PATCHES_PER_SHARD = 64        # patches por arquivo .tfrecord
INT16_SCALE       = 10_000.0  # divide int16 [0, 10000] → float32 [0, 1]

# Filtros opcionais (None = processa tudo que estiver em NPY_DIR)
FILTER_YEARS   = None   # ex.: [2020, 2021, 2022]
FILTER_REGIONS = None   # ex.: ['0001', '0002']

# ==============================================================================
# 3. ESPECIFICAÇÃO DE LEITURA (para uso externo no pipeline de inferência)
# ==============================================================================

FEATURE_SPEC = {
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
# 4. FUNÇÕES — SERIALIZAÇÃO
# ==============================================================================

def _bytes(v: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))

def _int64(v) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list(v)))

def _float(v) -> tf.train.Feature:
    return tf.train.Feature(float_list=tf.train.FloatList(value=list(v)))


def make_example(arr_hwc: np.ndarray, meta: dict) -> tf.train.Example:
    """Serializa um patch (H, W, C) int16 + metadados → tf.train.Example."""
    arr_f32 = (arr_hwc.astype(np.float32) / INT16_SCALE).clip(0.0, 1.0)
    return tf.train.Example(features=tf.train.Features(feature={
        'patch/data':     _bytes(arr_f32.tobytes()),
        'patch/shape':    _int64(arr_f32.shape),
        'meta/region_id': _bytes(meta['region_id'].encode()),
        'meta/year':      _int64([meta['year']]),
        'meta/row':       _int64([meta['row']]),
        'meta/col':       _int64([meta['col']]),
        'meta/transform': _float(meta['transform']),
        'meta/crs':       _bytes(meta['crs'].encode()),
    }))


def decode_example(proto):
    """
    Decodifica um Example do TFRecord para uso em tf.data.Dataset.

    Uso:
      dataset = tf.data.TFRecordDataset(glob.glob('.../*.tfrecord'))
      dataset = dataset.map(decode_example)
      # retorna (patch [H, W, C] float32, meta dict)

    O transform GDAL no meta permite georreferenciar cada patch:
      import rasterio.transform
      transform = rasterio.transform.from_gdal(*meta['meta/transform'].numpy())
    """
    parsed = tf.io.parse_single_example(proto, FEATURE_SPEC)
    patch = tf.io.decode_raw(parsed['patch/data'], tf.float32)
    patch = tf.reshape(patch, parsed['patch/shape'])
    return patch, parsed

# ==============================================================================
# 5. FUNÇÕES — ESCRITA
# ==============================================================================

def write_shard(npy_paths: list, shard_path: Path) -> tuple:
    """
    Escreve um shard TFRecord com os patches da lista npy_paths.
    Retorna (n_escritos, n_falhos).
    """
    written = failed = 0
    with tf.io.TFRecordWriter(str(shard_path)) as writer:
        for npy_path in npy_paths:
            json_path = npy_path.with_suffix('.json')
            try:
                arr  = np.load(npy_path)
                meta = json.loads(json_path.read_text(encoding='utf-8'))
                writer.write(make_example(arr, meta).SerializeToString())
                written += 1
            except Exception as exc:
                log.error(f"  Falha em {npy_path.name}: {exc}")
                failed += 1
    # Remove shard vazio se todos os patches falharam
    if written == 0 and shard_path.exists():
        shard_path.unlink()
    return written, failed


def convert_region_year(region_dir: Path, year_dir: Path) -> dict:
    """
    Converte todos os patches de uma pasta <region>/<year> em shards TFRecord.
    Shards já existentes são pulados (resume automático).
    Retorna estatísticas {total, written, skipped, failed}.
    """
    npy_files = sorted(year_dir.glob('patch_*.npy'))
    if not npy_files:
        return {'total': 0, 'written': 0, 'skipped': 0, 'failed': 0}

    out_dir = TFRECORD_DIR / region_dir.name / year_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    shards = [npy_files[i:i + PATCHES_PER_SHARD]
              for i in range(0, len(npy_files), PATCHES_PER_SHARD)]

    written = skipped = failed = 0

    for idx, shard_files in enumerate(shards):
        shard_path = out_dir / f"{region_dir.name}_{year_dir.name}_part_{idx:04d}.tfrecord"

        if shard_path.exists():
            skipped += len(shard_files)
            continue

        w, f = write_shard(shard_files, shard_path)
        written += w
        failed  += f

    return {'total': len(npy_files), 'written': written, 'skipped': skipped, 'failed': failed}

# ==============================================================================
# 6. PIPELINE PRINCIPAL
# ==============================================================================

def main():
    if not NPY_DIR.exists():
        log.error(f"Diretório NPY não encontrado: {NPY_DIR}")
        return

    TFRECORD_DIR.mkdir(parents=True, exist_ok=True)

    pairs = []
    for region_dir in sorted(NPY_DIR.iterdir()):
        if not region_dir.is_dir():
            continue
        if FILTER_REGIONS and region_dir.name not in FILTER_REGIONS:
            continue
        for year_dir in sorted(region_dir.iterdir()):
            if not year_dir.is_dir():
                continue
            if FILTER_YEARS and int(year_dir.name) not in FILTER_YEARS:
                continue
            pairs.append((region_dir, year_dir))

    log.info(f"Pares (região, ano) encontrados: {len(pairs)}")
    log.info(f"Saída: {TFRECORD_DIR}")

    total_written = total_skipped = total_failed = 0

    for region_dir, year_dir in tqdm(pairs, desc='convertendo', unit='dir',
                                     disable=not HAS_TQDM):
        log.info(f"  {region_dir.name}/{year_dir.name}")
        stats = convert_region_year(region_dir, year_dir)
        total_written += stats['written']
        total_skipped += stats['skipped']
        total_failed  += stats['failed']
        log.info(f"    total={stats['total']} | "
                 f"escritos={stats['written']} | "
                 f"pulados={stats['skipped']} | "
                 f"falhas={stats['failed']}")

    log.info(f"\nConcluído.")
    log.info(f"  Patches escritos : {total_written}")
    log.info(f"  Patches pulados  : {total_skipped}")
    log.info(f"  Patches com falha: {total_failed}")
    log.info(f"  Log: {LOG_FILE.resolve()}")


if __name__ == '__main__':
    main()
