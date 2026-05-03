#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training — Segmentação Fotovoltaica (NICFI/Planet)
===================================================
Treina UNet + ResNet50 na base TFRecord gerada pelo pipeline GEE.

Dataset esperado (gerado por split_tfrecord_dataset.py):
  <DATASET_ROOT>/train/  *.tfrecord.gz   (~80%)
  <DATASET_ROOT>/val/    *.tfrecord.gz   (~10%)
  <DATASET_ROOT>/test/   *.tfrecord.gz   (~10%)

Uso:
    python train_fotovoltaica.py <dataset_root> [--output <output_dir>]

Exemplos:
    python train_fotovoltaica.py /run/media/superuser/Almacen/imgDB/DATASET_FOTOVOLTAICA_NICFI_TFRECORDS
    python train_fotovoltaica.py /mnt/servidor/dados/tfrecords --output /mnt/servidor/modelos
    python train_fotovoltaica.py /dados/tfrecords --backbone xception --epochs 100
"""

import sys
import glob
import argparse
import datetime
from pathlib import Path
import pandas as pd
import tensorflow as tf

# Garante que o diretório src/ está no path para importar a factory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from segmentation_model_factory import (
    SegmentationModelFactory,
    hybrid_focal_loss,
    dice_coef    as factory_dice_coef,
    bce_dice_loss as factory_bce_dice_loss,
)

# ==============================================================================
# 1. CONFIGURAÇÕES
# ==============================================================================

DATASET_ROOT  = Path('/run/media/superuser/Almacen/imgDB/DATASET_FOTOVOLTAICA_NICFI_TFRECORDS')
OUTPUT_DIR    = DATASET_ROOT / 'TRAINING_OUTPUTS_FV'

# Arquitetura
SEGMENTATION_HEAD = 'unet'
BACKBONE_NAME     = 'resnet50'

# Bandas e patches
BANDS_LIST     = ['blue', 'green', 'red', 'nir', 'pvi', 'iia', 'ri', 'evi']
LABEL_KEY      = 'label'
FEATURES_KEYS  = BANDS_LIST + [LABEL_KEY]
PATCH_SIZE     = 256    # tamanho final após crop
RAW_PATCH_SIZE = 257    # saída do GEE: kernel rectangle(128,128) = 257x257
INPUT_SHAPE    = (PATCH_SIZE, PATCH_SIZE, len(BANDS_LIST))  # (256, 256, 8)

# Treinamento
EPOCHS         = 170
BATCH_SIZE     = 8
LEARNING_RATE  = 1e-4
DICE_THRESHOLD = 0.75
USE_ATTENTION  = False   # True → Heavyweight Focus U-Net com Focus Gates
LOSS_NAME      = 'bce_dice'   # 'bce_dice' | 'hybrid_focal'
FOCAL_GAMMA    = 1.25         # γ do Focus Gate e da hybrid_focal_loss

# Normalização: valores Planet NICFI e PVI exportados como Int16 escalados x10000
NORM_FACTOR = 10000.0


# ==============================================================================
# 2. GPU
# ==============================================================================

def configure_gpu() -> str:
    """Habilita memory growth e retorna o device string para treinamento."""
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print('[AVISO] Nenhuma GPU detectada — treinando em CPU.')
        return '/cpu:0'

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    print(f'[GPU] {len(gpus)} dispositivo(s) encontrado(s):')
    for i, g in enumerate(gpus):
        details = tf.config.experimental.get_device_details(g)
        print(f'  [{i}] {g.name}  —  {details.get("device_name", "n/a")}')

    return '/gpu:0'


# ==============================================================================
# 3. LOSSES E MÉTRICAS  (importadas da factory — package='RemoteSensing')
# ==============================================================================

# Aliases locais para usar nas chamadas compile() e nos callbacks
dice_coef     = factory_dice_coef
bce_dice_loss = factory_bce_dice_loss

# Mapa loss_name → função — usado em compile() e no nome dos artefatos
LOSS_MAP = {
    'bce_dice'    : bce_dice_loss,
    'hybrid_focal': hybrid_focal_loss,
}


# ==============================================================================
# 4. CALLBACKS
# ==============================================================================

class SimpleDiceThreshold(tf.keras.callbacks.Callback):
    """Para o treino ao atingir `threshold` de dice_coef na validação."""

    def __init__(self, threshold=0.75, monitor='val_dice_coef', verbose=1):
        super().__init__()
        self.threshold = threshold
        self.monitor   = monitor
        self.verbose   = verbose

    def on_epoch_end(self, epoch, logs=None):
        current = (logs or {}).get(self.monitor)
        if current and current >= self.threshold:
            if self.verbose:
                print(f'\n[SimpleDiceThreshold] {self.monitor}={current:.4f} '
                      f'>= {self.threshold}. Parando treinamento.')
            self.model.stop_training = True


def build_callbacks(checkpoint_path: str, dice_threshold: float = DICE_THRESHOLD) -> list:
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_dice_coef',
            mode='max',
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            verbose=1,
            min_lr=1e-8,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_dice_coef',
            patience=25,
            mode='max',
            min_delta=0.0001,
            restore_best_weights=True,
            verbose=1,
        ),
        SimpleDiceThreshold(threshold=dice_threshold, monitor='val_dice_coef', verbose=1),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(Path(checkpoint_path).parent / f'logs_{datetime.datetime.now().strftime("%Y%m%d_%H%M")}'),
            histogram_freq=0,
        ),
    ]


# ==============================================================================
# 5. PIPELINE DE DADOS (TFRecord)
# ==============================================================================

# GEE exporta neighborhoodToArray como float_list no protocolo TFRecord
FEATURE_DESCRIPTION = {
    key: tf.io.FixedLenFeature([RAW_PATCH_SIZE, RAW_PATCH_SIZE], tf.float32)
    for key in FEATURES_KEYS
}


def parse_tfrecord(example_proto):
    """
    Lê um exemplo e recorta de 257×257 → 256×256.
    O kernel rectangle(128,128) no GEE gera patches de 257×257.
    """
    parsed = tf.io.parse_single_example(example_proto, FEATURE_DESCRIPTION)
    return {
        key: tf.slice(parsed[key], [0, 0], [PATCH_SIZE, PATCH_SIZE])
        for key in FEATURES_KEYS
    }


def to_model_tuple(inputs: dict):
    """
    dict → (image, label)
    Normaliza int16 (0–10000) → float32 (0.0–1.0).
    Label binarizado para float32 {0.0, 1.0}.
    """
    img   = tf.stack([inputs[b] for b in BANDS_LIST], axis=-1)  # (256, 256, 5)
    label = tf.expand_dims(inputs[LABEL_KEY], axis=-1)           # (256, 256, 1)

    img   = tf.cast(img,   tf.float32) / NORM_FACTOR
    label = tf.cast(label, tf.float32)
    label = tf.clip_by_value(label, 0.0, 1.0)
    return img, label


def removeNan(image, label):
    image = tf.where(tf.math.is_nan(image), tf.zeros_like(image), image)
    label = tf.where(tf.math.is_nan(label), tf.zeros_like(label), label)
    return image, label


# Fração de patches vazios (sem nenhum painel) mantidos no treino.
# 0.25 = mantém 25% dos patches negativos; 1.0 = mantém todos.
EMPTY_PATCH_KEEP_RATE = 0.25

def filter_empty_patches(image, label):
    """
    Descarta patches completamente vazios com probabilidade (1 - EMPTY_PATCH_KEEP_RATE).
    Patches com pelo menos 1 pixel de painel são sempre mantidos.
    Evita que o modelo aprenda a prever fundo em tudo.
    """
    has_panel = tf.reduce_sum(label) > 0
    keep_empty = tf.random.uniform([]) < EMPTY_PATCH_KEEP_RATE
    return tf.logical_or(has_panel, keep_empty)


# ------------------------------------------------------------------
# Augmentation
# ------------------------------------------------------------------

def augment_spatial(image, label):
    """Translação aleatória via pad+crop (64 px em cada direção).
    label já chega como (256,256,1) — não adicionar newaxis.
    """
    pad = 64
    padded_img   = tf.pad(image, [[pad, pad], [pad, pad], [0, 0]])
    padded_label = tf.pad(label, [[pad, pad], [pad, pad], [0, 0]])

    rx = tf.random.uniform([], 0, pad * 2 + 1, dtype=tf.int32)
    ry = tf.random.uniform([], 0, pad * 2 + 1, dtype=tf.int32)

    img_crop   = tf.slice(padded_img,   [rx, ry, 0], [PATCH_SIZE, PATCH_SIZE, len(BANDS_LIST)])
    label_crop = tf.slice(padded_label, [rx, ry, 0], [PATCH_SIZE, PATCH_SIZE, 1])
    return img_crop, label_crop   # mantém (256,256,1)


def augment_flip_rotate(image, label):
    """Flip horizontal/vertical e rotação 90° aleatória (sincronizados).
    label já chega como (256,256,1) — concat direto sem newaxis.
    """
    combined = tf.concat([image, label], axis=-1)   # (256, 256, 6)

    combined = tf.image.random_flip_left_right(combined)
    combined = tf.image.random_flip_up_down(combined)
    combined = tf.image.rot90(combined, tf.random.uniform([], 0, 4, dtype=tf.int32))

    img_aug   = combined[:, :, :len(BANDS_LIST)]
    label_aug = combined[:, :, len(BANDS_LIST):]    # mantém (256,256,1)
    return img_aug, label_aug


def augment_radiometric(image, label):
    """Brilho, contraste e ruído gaussiano — apenas na imagem, nunca no label."""
    # Brilho: deslocamento aleatório ±0.1
    image = image + tf.random.uniform([], -0.10, 0.10)
    # Contraste: escala aleatória em [0.8, 1.2]
    image = image * tf.random.uniform([], 0.80, 1.20)
    # Ruído gaussiano leve
    image = image + tf.random.normal(tf.shape(image), stddev=0.02)
    # Mantém no intervalo válido [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def augment_all(image, label):
    image, label = augment_flip_rotate(image, label)
    image, label = augment_spatial(image, label)
    image, label = augment_radiometric(image, label)
    return image, label


# ------------------------------------------------------------------
# Dataset builder
# ------------------------------------------------------------------

def build_dataset(split: str, batch_size: int, augment: bool = False,
                  root: Path = None) -> tf.data.Dataset:
    """
    Constrói tf.data.Dataset a partir dos TFRecords de um split.
    Augmentation apenas no split 'train'.
    """
    base    = root if root is not None else DATASET_ROOT
    pattern = str(base / split / '*.tfrecord.gz')
    files   = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f'Nenhum TFRecord encontrado em: {DATASET_ROOT / split}\n'
            f'Execute split_tfrecord_dataset.py primeiro.'
        )
    print(f'  [{split:5s}] {len(files)} arquivo(s)')

    ds = tf.data.TFRecordDataset(files,
                                 compression_type='GZIP',
                                 num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(parse_tfrecord,  num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(to_model_tuple,  num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(removeNan,       num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        # Filtra patches vazios no treino para não viciar o modelo em "prever fundo"
        ds = ds.filter(filter_empty_patches)
        ds = ds.map(augment_all, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.cache()  # fixa a seleção
        ds = ds.repeat() 
        ds = ds.shuffle(buffer_size=500, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ==============================================================================
# 6. MAIN
# ==============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Treina modelo de segmentação fotovoltaica com TFRecords NICFI.'
    )
    p.add_argument(
        'dataset_root', type=Path,
        help='Pasta raiz com subpastas train/, val/, test/ contendo os .tfrecord.gz',
    )
    p.add_argument(
        '--output', type=Path,
        default=None,
        help='Pasta de saída para modelos e histórico (default: <dataset_root>/TRAINING_OUTPUTS_FV)',
    )
    p.add_argument(
        '--head', default=SEGMENTATION_HEAD,
        choices=['unet', 'deeplabv3plus', 'pspnet'],
        help='Arquitetura de segmentação (default: %(default)s)',
    )
    p.add_argument(
        '--backbone', default=BACKBONE_NAME,
        help='Backbone encoder (default: %(default)s)',
    )
    p.add_argument(
        '--epochs', type=int, default=EPOCHS,
        help='Número máximo de épocas (default: %(default)d)',
    )
    p.add_argument(
        '--batch', type=int, default=BATCH_SIZE,
        help='Batch size (default: %(default)d)',
    )
    p.add_argument(
        '--lr', type=float, default=LEARNING_RATE,
        help='Learning rate inicial (default: %(default)g)',
    )
    p.add_argument(
        '--dice-threshold', type=float, default=DICE_THRESHOLD,
        help='Para o treino ao atingir este Dice na validação (default: %(default)g)',
    )
    p.add_argument(
        '--attention', action='store_true', default=USE_ATTENTION,
        help='Usa Heavyweight Focus U-Net com Focus Gates (canal + espacial + filtro focal)',
    )
    p.add_argument(
        '--loss', default=LOSS_NAME, choices=list(LOSS_MAP),
        help='Função de perda: bce_dice (padrão) ou hybrid_focal (recomendada com --attention)',
    )
    p.add_argument(
        '--gamma', type=float, default=FOCAL_GAMMA,
        help='γ do Focus Gate e da hybrid_focal_loss (default: %(default)g)',
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    dataset_root = args.dataset_root
    output_dir   = args.output if args.output is not None \
                   else dataset_root / 'TRAINING_OUTPUTS_FV'
    head         = args.head
    backbone     = args.backbone
    epochs       = args.epochs
    batch_size   = args.batch
    lr           = args.lr
    dice_thr     = args.dice_threshold
    use_attention = args.attention
    loss_name    = args.loss
    focal_gamma  = args.gamma
    input_shape  = (PATCH_SIZE, PATCH_SIZE, len(BANDS_LIST))

    if not dataset_root.exists():
        raise FileNotFoundError(f'dataset_root não encontrado: {dataset_root}')

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    device    = configure_gpu()

    attn_tag = '_focus' if use_attention else ''
    print(f'\nDataset root : {dataset_root}')
    print(f'Output dir   : {output_dir}')
    print(f'Arquitetura  : {head.upper()}{attn_tag} + {backbone}  |  loss={loss_name}  γ={focal_gamma}')

    # ---- Datasets ----
    print('\nConstruindo datasets...')
    train_ds = build_dataset('train', batch_size, augment=True,  root=dataset_root)
    val_ds   = build_dataset('val',   batch_size, augment=False, root=dataset_root)

    # ---- Modelo ----
    print(f'\nConstruindo: {head.upper()}{attn_tag} + {backbone}  |  input={input_shape}')
    factory = SegmentationModelFactory(
        segmentation_head=head,
        backbone_name=backbone,
        input_shape=input_shape,
        use_attention=use_attention,
        focal_gamma=focal_gamma,
    )
    model = factory.build()
    print(f'Parâmetros: {model.count_params():,}')

    # ---- Compilação ----
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=LOSS_MAP[loss_name],
        metrics=[
            dice_coef,
            tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5, name='iou'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ],
    )

    # ---- Callbacks ----
    checkpoint_path = str(output_dir / f'best_{head}{attn_tag}_{backbone}_{timestamp}.keras')
    callbacks_list  = build_callbacks(checkpoint_path, dice_thr)
    print(f'\nCheckpoint → {checkpoint_path}')

    # ---- Treinamento ----
    print(f'\nIniciando treinamento  |  epochs={epochs}  batch={batch_size}  lr={lr}')
    with tf.device(device):
        result = model.fit(
            train_ds,
            epochs=epochs,
            validation_data=val_ds,
            callbacks=callbacks_list,
            verbose=1,
        )

    # ---- Histórico ----
    csv_path = output_dir / f'history_{head}{attn_tag}_{backbone}_{timestamp}.csv'
    df = pd.DataFrame(result.history)
    df['epoch'] = df.index + 1
    df.to_csv(csv_path, index=False)
    print(f'\nHistórico salvo: {csv_path}')
    print('Treinamento finalizado.')


if __name__ == '__main__':
    main()
