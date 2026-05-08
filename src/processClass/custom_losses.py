#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
custom_losses.py
================
Define e registra todos os custom objects Keras usados nos modelos de
fotovoltaica — com os packages exatos usados no treinamento (Colab).

  - dice_coef / dice_loss          → package='Custom'   (célula 24 do Colab)
  - focal_tversky_loss             → package='RemoteSensing'
  - boundary_loss                  → package='RemoteSensing'
  - focal_tversky_boundary_loss    → package='RemoteSensing'

Importar este módulo ANTES de tf.keras.models.load_model() garante que
todos os objetos customizados estejam registrados no namespace do Keras.

Uso:
  from custom_losses import build_custom_objects
  model = tf.keras.models.load_model(path, custom_objects=build_custom_objects())
"""

import tensorflow as tf
import keras
import keras.ops as kops

# ── dice_coef / dice_loss ─────────────────────────────────────────────────────
# package='Custom' — igual ao usado no Colab (célula 24).
# O modelo salvo referencia 'Custom>dice_coef', não 'RemoteSensing>dice_coef'.

@keras.utils.register_keras_serializable(package='Custom')
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = kops.reshape(kops.cast(y_true, 'float32'), [-1])
    y_pred_f = kops.reshape(kops.cast(y_pred, 'float32'), [-1])
    intersection = kops.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (kops.sum(y_true_f) + kops.sum(y_pred_f) + smooth)


@keras.utils.register_keras_serializable(package='Custom')
def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


# ── losses do treinamento principal ───────────────────────────────────────────

@keras.utils.register_keras_serializable(package='RemoteSensing')
def focal_tversky_loss(y_true, y_pred, alpha=0.3, beta=0.7, gamma=1.25, smooth=1e-6):
    """Focal Tversky Loss — alpha=0.3 beta=0.7 foca em Recall (reduz FN)."""
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
    tversky  = focal_tversky_loss(y_true, y_pred, alpha=alpha, beta=beta,
                                   gamma=gamma, smooth=smooth)
    boundary = boundary_loss(y_true, y_pred, smooth=smooth)
    return tversky + boundary_weight * boundary


# ── ResizeLike (da factory) ───────────────────────────────────────────────────
# Reexportado aqui para que os scripts só precisem importar custom_losses.

try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from segmentation_model_factory import ResizeLike, bce_dice_loss, hybrid_focal_loss
except Exception:
    ResizeLike      = None
    bce_dice_loss   = None
    hybrid_focal_loss = None


def build_custom_objects() -> dict:
    """
    Retorna dicionário pronto para passar a tf.keras.models.load_model().
    Inclui todos os objetos customizados usados nos modelos de fotovoltaica.
    """
    objs = {
        'dice_coef':                   dice_coef,
        'dice_loss':                   dice_loss,
        'focal_tversky_loss':          focal_tversky_loss,
        'boundary_loss':               boundary_loss,
        'focal_tversky_boundary_loss': focal_tversky_boundary_loss,
    }
    if ResizeLike is not None:
        objs['ResizeLike'] = ResizeLike
    if bce_dice_loss is not None:
        objs['bce_dice_loss'] = bce_dice_loss
    if hybrid_focal_loss is not None:
        objs['hybrid_focal_loss'] = hybrid_focal_loss
    return objs
