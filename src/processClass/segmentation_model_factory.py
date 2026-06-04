#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SegmentationModelFactory
========================
Factory para criação, compilação e configuração de modelos de segmentação
semântica para sensoriamento remoto (5 canais de entrada por padrão).

Heads suportadas : UNet, DeepLabV3+, PSPNet, FPN
Backbones        : InceptionV3, ResNet50/101/152, ResNext50 (→ ResNet50),
                   MobileNet, Xception, EfficientNetB0/B3/B7, InceptionResNetV2

NOTA: Todos os backbones são inicializados com weights=None (sem ImageNet),
      pois a entrada tem 5 bandas espectrais (incompatível com pesos RGB).

Compatível com Keras 3 (usa keras.ops em vez de tf.* nos modelos).
"""

import keras
from keras import layers, Model
import keras.ops as kops


# ==============================================================================
# 1. CUSTOM OBJECTS
# ==============================================================================

@keras.utils.register_keras_serializable(package='RemoteSensing')
def dice_coef(y_true, y_pred, smooth: float = 1e-6):
    """F1-Score diferenciável para segmentação binária."""
    y_true_f = kops.cast(kops.reshape(y_true, [-1]), 'float32')
    y_pred_f = kops.cast(kops.reshape(y_pred, [-1]), 'float32')
    intersection = kops.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        kops.sum(y_true_f) + kops.sum(y_pred_f) + smooth
    )


@keras.utils.register_keras_serializable(package='RemoteSensing')
def dice_loss(y_true, y_pred):
    """1 − dice_coef."""
    return 1.0 - dice_coef(y_true, y_pred)


@keras.utils.register_keras_serializable(package='RemoteSensing')
def contrastive_loss(y_true, y_pred, margin: float = 1.0):
    """Contrastive Loss estilo DASNet."""
    y_true = kops.cast(y_true, 'float32')
    y_pred = kops.cast(y_pred, 'float32')
    loss_sim   = y_true * kops.square(y_pred)
    loss_disim = (1.0 - y_true) * kops.square(kops.maximum(margin - y_pred, 0.0))
    return kops.mean(0.5 * (loss_sim + loss_disim))


@keras.utils.register_keras_serializable(package='RemoteSensing')
def bce_dice_loss(y_true, y_pred):
    """Perda híbrida BCE + Dice."""
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)


@keras.utils.register_keras_serializable(package='RemoteSensing')
def hybrid_focal_loss(y_true, y_pred, alpha: float = 0.25,
                      gamma: float = 2.0, smooth: float = 1e-6):
    """
    Hybrid Focal Loss (HFL) = Focal BCE + Focal Dice.

    Indicada para dados com forte desequilíbrio de classes (painéis solares ~1–3 %
    dos pixels). O fator (1 - p_t)^γ concentra o gradiente nas amostras difíceis
    e suprime a contribuição das amostras fáceis (fundo dominante).

    Focal BCE:  -α_t * (1-p_t)^γ * log(p_t)
    Focal Dice: (1 - Dice)^γ          ← aplica o filtro focal ao Dice
    """
    y_true = kops.cast(y_true, 'float32')
    y_pred = kops.cast(y_pred, 'float32')
    y_pred = kops.clip(y_pred, 1e-7, 1.0 - 1e-7)

    # ── Focal BCE ────────────────────────────────────────────────────────────
    bce      = -(y_true * kops.log(y_pred) + (1 - y_true) * kops.log(1 - y_pred))
    p_t      = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    alpha_t  = y_true * alpha + (1 - y_true) * (1 - alpha)
    focal_w  = alpha_t * kops.power(1.0 - p_t, gamma)
    focal_bce = kops.mean(focal_w * bce)

    # ── Focal Dice ────────────────────────────────────────────────────────────
    y_true_f = kops.reshape(y_true, [-1])
    y_pred_f = kops.reshape(y_pred, [-1])
    inter    = kops.sum(y_true_f * y_pred_f)
    d        = (2.0 * inter + smooth) / (kops.sum(y_true_f) + kops.sum(y_pred_f) + smooth)
    focal_dice = kops.power(1.0 - d, gamma)

    return focal_bce + focal_dice


# ------------------------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------------------------

class SimpleDiceThreshold(keras.callbacks.Callback):
    """Para o treino ao atingir `threshold` de dice_coef na validação."""

    def __init__(self, threshold: float = 0.85, monitor: str = 'val_dice_coef'):
        super().__init__()
        self.threshold = threshold
        self.monitor   = monitor

    def on_epoch_end(self, epoch, logs=None):
        val_dice = (logs or {}).get(self.monitor, 0.0)
        if val_dice >= self.threshold:
            print(f"\n[SimpleDiceThreshold] Epoch {epoch + 1}: "
                  f"{self.monitor}={val_dice:.4f} >= {self.threshold}. Parando.")
            self.model.stop_training = True


def get_default_callbacks(checkpoint_path: str = 'best_model.keras',
                          dice_threshold: float = 0.95) -> list:
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, monitor='val_dice_coef',
            mode='max', save_best_only=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_dice_coef', mode='max', patience=20,
            restore_best_weights=True, verbose=1,
        ),
        SimpleDiceThreshold(threshold=dice_threshold, monitor='val_dice_coef'),
    ]


# ==============================================================================
# 2. LAYER AUXILIAR — ResizeLike
# ==============================================================================

class ResizeLike(keras.layers.Layer):
    """
    Redimensiona `x` para o mesmo tamanho espacial de `target` via bilinear.

    Usa tf.image.resize diretamente para evitar incompatibilidades entre
    versões do Keras 3 (kops.image.resize traduz parâmetros de forma
    inconsistente entre TF 2.x e JAX backends).

    Uso:  out = ResizeLike()([x, target])
    """
    def call(self, inputs):
        import tensorflow as tf
        x, target = inputs
        target_shape = tf.shape(target)
        return tf.image.resize(x, [target_shape[1], target_shape[2]])

    def compute_output_shape(self, input_shape):
        x_shape, target_shape = input_shape
        return (x_shape[0], target_shape[1], target_shape[2], x_shape[3])


# ==============================================================================
# 3. MAPEAMENTOS
# ==============================================================================

_BACKBONE_ALIASES: dict[str, str] = {
    'inceptionv3'      : 'inceptionv3',
    'resnet50'         : 'resnet50',
    'resnet101'        : 'resnet101',
    'resnet152'        : 'resnet152',
    'resnext50'        : 'resnext50',
    'mobilenet'        : 'mobilenet',
    'xception'         : 'xception',
    'efficientnetb0'   : 'efficientnetb0',
    'efficientnetb3'   : 'efficientnetb3',
    'efficientnetb7'   : 'efficientnetb7',
    'inceptionresnetv2': 'inceptionresnetv2',
    'inceptionresnet'  : 'inceptionresnetv2',
}

_SUPPORTED_HEADS: set[str] = {'unet', 'deeplabv3plus', 'pspnet', 'fpn'}

# backbone → (AppClass, low_layer, medium_layer, high_layer)
# low    ≈ stride  4 — detalhes finos (bordas, texturas)
# medium ≈ stride  8 — features semânticas intermediárias
# high   ≈ stride 16 — features semânticas profundas (sinal de gating no Focus Gate)
_KERAS_BACKBONE_MAP: dict[str, tuple] = {
    'inceptionv3'      : (keras.applications.InceptionV3,       'mixed2',                    'mixed5',                    'mixed7'),
    'resnet50'         : (keras.applications.ResNet50,           'conv2_block3_out',           'conv3_block4_out',           'conv4_block6_out'),
    'resnet101'        : (keras.applications.ResNet101,          'conv2_block3_out',           'conv3_block4_out',           'conv4_block23_out'),
    'resnet152'        : (keras.applications.ResNet152,          'conv2_block3_out',           'conv3_block8_out',           'conv4_block36_out'),
    'resnext50'        : (keras.applications.ResNet50,           'conv2_block3_out',           'conv3_block4_out',           'conv4_block6_out'),
    'mobilenet'        : (keras.applications.MobileNet,          'conv_pw_3_relu',             'conv_pw_5_relu',             'conv_pw_11_relu'),
    'xception'         : (keras.applications.Xception,           'block3_sepconv2_act',        'block9_sepconv2_act',        'block13_sepconv2_act'),
    'efficientnetb0'   : (keras.applications.EfficientNetB0,     'block3a_expand_activation',  'block4a_expand_activation',  'block6a_expand_activation'),
    'efficientnetb3'   : (keras.applications.EfficientNetB3,     'block3a_expand_activation',  'block4a_expand_activation',  'block6a_expand_activation'),
    'efficientnetb7'   : (keras.applications.EfficientNetB7,     'block3a_expand_activation',  'block5a_expand_activation',  'block7a_expand_activation'),
    'inceptionresnetv2': (keras.applications.InceptionResNetV2,  'activation_3',               'block17_10_ac',              'block17_20_ac'),
}


# ==============================================================================
# 4. FACTORY
# ==============================================================================

class SegmentationModelFactory:
    """
    Factory para modelos de segmentação semântica em sensoriamento remoto.

    Parameters
    ----------
    segmentation_head : str
        'unet', 'deeplabv3plus' ou 'pspnet'
    backbone_name : str | None
        Backbone encoder. Se None, usa arquitetura vanilla.
        Todos os backbones usam weights=None (5 bandas, sem ImageNet).
    input_shape : tuple
        (H, W, C) — padrão (256, 256, 5).
    """

    def __init__(self,
                 segmentation_head: str,
                 backbone_name: str | None = None,
                 input_shape: tuple = (256, 256, 5),
                 use_attention: bool = False,
                 focal_gamma: float = 1.25):

        head = segmentation_head.lower().replace('-', '').replace('_', '')
        if head not in _SUPPORTED_HEADS:
            raise ValueError(f"segmentation_head='{segmentation_head}' inválido. "
                             f"Opções: {sorted(_SUPPORTED_HEADS)}")

        if backbone_name is not None:
            bb = backbone_name.lower().replace('-', '').replace('_', '')
            if bb not in _BACKBONE_ALIASES:
                raise ValueError(f"backbone_name='{backbone_name}' inválido. "
                                 f"Opções: {sorted(_BACKBONE_ALIASES)}")
            backbone_name = _BACKBONE_ALIASES[bb]

        self.segmentation_head = head
        self.backbone_name     = backbone_name
        self.input_shape       = input_shape
        self.use_attention     = use_attention
        self.focal_gamma       = focal_gamma
        self.model: Model | None = None

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def build(self) -> Model:
        builders = {
            'unet'         : self._build_unet,
            'deeplabv3plus': self._build_deeplabv3plus,
            'pspnet'       : self._build_pspnet,
            'fpn'          : self._build_fpn,
        }
        self.model = builders[self.segmentation_head]()
        return self.model

    def compile_model(self, learning_rate: float = 1e-4,
                      loss: str = 'bce_dice') -> None:
        """
        Compila o modelo.

        loss:
          'bce_dice'     — BCE + Dice (padrão)
          'hybrid_focal' — Hybrid Focal Loss, recomendada com use_attention=True
                           (lida melhor com o forte desequilíbrio de classes)
        """
        if self.model is None:
            raise RuntimeError("Chame build() antes de compile_model().")
        loss_fn = hybrid_focal_loss if loss == 'hybrid_focal' else bce_dice_loss
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
            loss=loss_fn,
            metrics=['accuracy', dice_coef],
        )

    def get_callbacks(self, checkpoint_path: str = 'best_model.keras',
                      dice_threshold: float = 0.85) -> list:
        return get_default_callbacks(checkpoint_path, dice_threshold)

    # ------------------------------------------------------------------
    # Head builders
    # ------------------------------------------------------------------

    def _build_unet(self) -> Model:
        if self.use_attention:
            return (self._focus_unet_vanilla() if self.backbone_name is None
                    else self._focus_unet_with_backbone())
        return (self._vanilla_unet() if self.backbone_name is None
                else self._unet_with_backbone())

    def _build_pspnet(self) -> Model:
        return self._vanilla_pspnet() if self.backbone_name is None \
               else self._pspnet_with_backbone()

    def _build_deeplabv3plus(self) -> Model:
        return self._deeplabv3plus_with_encoder(use_backbone=False) \
               if self.backbone_name is None \
               else self._deeplabv3plus_with_encoder(use_backbone=True)

    def _build_fpn(self) -> Model:
        return (self._vanilla_fpn() if self.backbone_name is None
                else self._fpn_with_backbone())

    # ------------------------------------------------------------------
    # Backbone helper (sem pesos pré-treinados — 5 canais)
    # ------------------------------------------------------------------

    def _get_backbone_features(self, inputs):
        """
        Extrai (low, medium, high) features do backbone (3 escalas).
        Sempre weights=None: entrada tem N bandas espectrais, incompatível com ImageNet.

        low    ≈ stride  4 — detalhes finos
        medium ≈ stride  8 — features intermediárias
        high   ≈ stride 16 — features semânticas (sinal de gating no Focus Gate)
        """
        AppClass, low_name, med_name, high_name = _KERAS_BACKBONE_MAP[self.backbone_name]
        base = AppClass(include_top=False, weights=None, input_tensor=inputs)
        base.trainable = True
        n_ch = self.input_shape[-1]
        print(f'[Backbone] {self.backbone_name}  weights=None  ({n_ch} canais)')
        return (base.get_layer(low_name).output,
                base.get_layer(med_name).output,
                base.get_layer(high_name).output)

    # ------------------------------------------------------------------
    # Focus Gate — módulos de atenção (Keras 3, sem tf.*)
    # ------------------------------------------------------------------

    @staticmethod
    def _channel_attention(x, ratio: int = 8):
        """
        Atenção de Canal (What): seleciona quais canais são relevantes.
        Dual-path Avg + Max Pool → MLP compartilhada → sigmoid.
        Saída: (B, H, W, C) com canais recalibrados.
        """
        C = x.shape[-1] or 1
        r = max(1, C // ratio)
        avg = layers.GlobalAveragePooling2D(keepdims=True)(x)   # (B,1,1,C)
        mx  = layers.GlobalMaxPooling2D(keepdims=True)(x)       # (B,1,1,C)
        fc1 = layers.Dense(r, activation='relu',
                           kernel_initializer='he_normal',
                           use_bias=False)
        fc2 = layers.Dense(C, kernel_initializer='he_normal',
                           use_bias=False)
        mask = layers.Activation('sigmoid')(
                   layers.Add()([fc2(fc1(avg)), fc2(fc1(mx))]))
        return layers.Multiply()([x, mask])

    @staticmethod
    def _spatial_attention(x):
        """
        Atenção Espacial (Where): refina a localização precisa dos objetos.
        Avg + Max ao longo do eixo de canal → Conv7x7 → sigmoid.
        Saída: (B, H, W, C) com regiões espaciais recalibradas.
        """
        avg = kops.mean(x, axis=-1, keepdims=True)   # (B,H,W,1)
        mx  = kops.max(x, axis=-1, keepdims=True)    # (B,H,W,1)
        concat = layers.Concatenate(axis=-1)([avg, mx])
        mask = layers.Conv2D(1, 7, padding='same', activation='sigmoid',
                             use_bias=False)(concat)
        return layers.Multiply()([x, mask])

    def _focus_gate(self, skip, gate):
        """
        Focus Gate (FG): combina atenção de canal + espacial com filtro focal γ.

          1. gate → Conv1×1 para alinhar canais → ResizeLike para alinhar resolução
          2. combined = ReLU(skip + gate_resized)
          3. ch  = channel_attention(combined)   ← "What"
          4. sp  = spatial_attention(combined)   ← "Where"
          5. coeff = sigmoid( (ch * sp)^γ )      ← filtro focal suprime fundo
          6. return skip * coeff

        γ > 1 amplifica contraste: regiões com alta atenção mantidas,
        ruído de fundo (baixa atenção) suprimido exponencialmente.
        """
        C = skip.shape[-1] or 32
        g = layers.Conv2D(C, 1, padding='same', use_bias=False)(gate)
        g = ResizeLike()([g, skip])
        combined = layers.Activation('relu')(layers.Add()([skip, g]))
        ch    = self._channel_attention(combined)
        sp    = self._spatial_attention(combined)
        coeff = kops.power(layers.Multiply()([ch, sp]), self.focal_gamma)
        coeff = layers.Activation('sigmoid')(coeff)
        return layers.Multiply()([skip, coeff])

    # ------------------------------------------------------------------
    # Utilitário conv
    # ------------------------------------------------------------------

    @staticmethod
    def _conv_bn_relu(x, filters, kernel=3, **kwargs):
        x = layers.Conv2D(filters, kernel, padding='same',
                          activation='relu', **kwargs)(x)
        return layers.BatchNormalization()(x)

    @staticmethod
    def _fpn_lateral(x, filters: int = 256):
        """Projeção lateral 1×1 sem ativação — alinha canais na pirâmide FPN."""
        x = layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
        return layers.BatchNormalization()(x)

    # ------------------------------------------------------------------
    # U-Net vanilla
    # ------------------------------------------------------------------

    def _vanilla_unet(self) -> Model:
        inputs = keras.Input(shape=self.input_shape, name='input')

        def _enc(x, f):
            x = self._conv_bn_relu(x, f)
            x = self._conv_bn_relu(x, f)
            return x

        def _dec(x, skip, f, drop=0.3):
            x = layers.Conv2DTranspose(f, 2, strides=2, padding='same')(x)
            x = layers.Concatenate()([x, skip])
            x = self._conv_bn_relu(x, f)
            x = layers.Dropout(drop)(x)
            x = self._conv_bn_relu(x, f)
            return x

        c1 = _enc(inputs, 64);  p1 = layers.MaxPooling2D()(c1)
        c2 = _enc(p1,     128); p2 = layers.MaxPooling2D()(c2)
        c3 = _enc(p2,     256); p3 = layers.MaxPooling2D()(c3)
        c4 = _enc(p3,     512); p4 = layers.MaxPooling2D()(c4)
        b  = _enc(p4,    1024); b  = layers.Dropout(0.3)(b)
        u6 = _dec(b,  c4, 512)
        u7 = _dec(u6, c3, 256)
        u8 = _dec(u7, c2, 128)
        u9 = _dec(u8, c1,  64)

        out = layers.Conv2D(1, 1, activation='sigmoid', name='output')(u9)
        return Model(inputs, out, name='unet_vanilla')

    # ------------------------------------------------------------------
    # U-Net com backbone
    # ------------------------------------------------------------------

    def _unet_with_backbone(self) -> Model:
        inputs = keras.Input(shape=self.input_shape, name='input')
        low, medium, high = self._get_backbone_features(inputs)

        # high → mesmo tamanho de medium
        x = self._conv_bn_relu(high, 256, kernel=1)
        x = ResizeLike(name='upsample_high_to_med')([x, medium])
        medium_p = self._conv_bn_relu(medium, 256, kernel=1)
        x = layers.Concatenate()([x, medium_p])
        x = self._conv_bn_relu(x, 256)
        x = layers.Dropout(0.3)(x)
        x = self._conv_bn_relu(x, 256)

        # medium → mesmo tamanho de low
        x = ResizeLike(name='upsample_med_to_low')([x, low])
        low_p = self._conv_bn_relu(low, 128, kernel=1)
        x = layers.Concatenate()([x, low_p])
        x = self._conv_bn_relu(x, 128)
        x = layers.Dropout(0.2)(x)
        x = self._conv_bn_relu(x, 128)

        # upsample para resolução original
        x = ResizeLike(name='upsample_to_input')([x, inputs])
        x = self._conv_bn_relu(x, 64)
        x = self._conv_bn_relu(x, 64)

        out = layers.Conv2D(1, 1, activation='sigmoid', name='output')(x)
        return Model(inputs, out, name=f'unet_{self.backbone_name}')

    # ------------------------------------------------------------------
    # Heavyweight Focus U-Net — vanilla (4 níveis + Focus Gates)
    # ------------------------------------------------------------------

    def _focus_unet_vanilla(self) -> Model:
        """
        Focus U-Net vanilla (sem backbone externo).
        Idêntico ao _vanilla_unet mas cada skip connection passa por um
        Focus Gate antes da concatenação no decoder.

        γ (focal_gamma) controla a supressão de fundo:
          γ=1.0 → atenção linear (sem filtro focal)
          γ=1.25 → suprime levemente o ruído
          γ=2.0 → supressão forte (útil quando fundo domina muito)
        """
        inputs = keras.Input(shape=self.input_shape, name='input')

        # ── Encoder ──────────────────────────────────────────────────
        def _enc(x, f):
            x = self._conv_bn_relu(x, f)
            x = self._conv_bn_relu(x, f)
            return x

        c1 = _enc(inputs, 64);  p1 = layers.MaxPooling2D()(c1)
        c2 = _enc(p1,     128); p2 = layers.MaxPooling2D()(c2)
        c3 = _enc(p2,     256); p3 = layers.MaxPooling2D()(c3)
        c4 = _enc(p3,     512); p4 = layers.MaxPooling2D()(c4)
        b  = _enc(p4,    1024); b  = layers.Dropout(0.3)(b)

        # ── Decoder com Focus Gates ───────────────────────────────────
        # O sinal de gating de cada nível vem do bloco decoder mais profundo,
        # que carrega mais semântica — mesmo princípio do attention U-Net.
        def _dec_focus(x, skip, f, gate_signal, drop=0.3):
            fg = self._focus_gate(skip, gate_signal)
            x  = layers.Conv2DTranspose(f, 2, strides=2, padding='same')(x)
            x  = layers.Concatenate()([x, fg])
            x  = self._conv_bn_relu(x, f)
            x  = layers.Dropout(drop)(x)
            x  = self._conv_bn_relu(x, f)
            return x

        u6 = _dec_focus(b,  c4, 512, b)     # bottleneck gata c4
        u7 = _dec_focus(u6, c3, 256, u6)    # u6 gata c3
        u8 = _dec_focus(u7, c2, 128, u7)    # u7 gata c2
        u9 = _dec_focus(u8, c1,  64, u8)    # u8 gata c1

        out = layers.Conv2D(1, 1, activation='sigmoid', name='output')(u9)
        return Model(inputs, out, name=f'focus_unet_vanilla_g{self.focal_gamma}')

    # ------------------------------------------------------------------
    # Heavyweight Focus U-Net — com backbone (3 níveis de skip)
    # ------------------------------------------------------------------

    def _focus_unet_with_backbone(self) -> Model:
        """
        Focus U-Net com backbone encoder (ResNet50, Xception, EfficientNetB3…).

        Arquitetura:
          backbone → low (stride~4) / medium (stride~8) / high (stride~16)

          high  → Focus Gate sobre medium  → decoder d_medium
          d_medium → Focus Gate sobre low   → decoder d_low
          d_low   → upsample até input      → saída

        O Focus Gate usa o sinal mais profundo disponível como gating,
        o que permite ao modelo selecionar canais (What) e regiões (Where)
        relevantes nas features de stride menor antes de concatená-las.
        """
        inputs = keras.Input(shape=self.input_shape, name='input')
        low, medium, high = self._get_backbone_features(inputs)

        # ── Bottleneck projection ─────────────────────────────────────
        b = self._conv_bn_relu(high, 512, kernel=1)
        b = self._conv_bn_relu(b, 512)
        b = layers.Dropout(0.3)(b)

        # ── Decoder nível 1: high gata medium ────────────────────────
        fg_med  = self._focus_gate(medium, b)
        x       = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(b)
        x       = ResizeLike()([x, fg_med])
        x       = layers.Concatenate()([x, fg_med])
        x       = self._conv_bn_relu(x, 256)
        x       = layers.Dropout(0.2)(x)
        d_med   = self._conv_bn_relu(x, 256)

        # ── Decoder nível 2: d_medium gata low ───────────────────────
        fg_low  = self._focus_gate(low, d_med)
        x       = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(d_med)
        x       = ResizeLike()([x, fg_low])
        x       = layers.Concatenate()([x, fg_low])
        x       = self._conv_bn_relu(x, 128)
        x       = layers.Dropout(0.2)(x)
        d_low   = self._conv_bn_relu(x, 128)

        # ── Upsample final para resolução do input ────────────────────
        x = ResizeLike(name='upsample_to_input')([d_low, inputs])
        x = self._conv_bn_relu(x, 64)
        x = self._conv_bn_relu(x, 64)

        out = layers.Conv2D(1, 1, activation='sigmoid', name='output')(x)
        name = f'focus_unet_{self.backbone_name}_g{self.focal_gamma}'
        return Model(inputs, out, name=name)

    # ------------------------------------------------------------------
    # PSPNet — Pyramid Pooling Module
    # ------------------------------------------------------------------

    def _ppm(self, feat, feat_stride: int, bins=(1, 2, 3, 6), filters=128):
        """
        Pyramid Pooling Module com pool_size estático calculado a partir
        de feat_stride (necessário para Keras 3 functional API).
        """
        H, W = self.input_shape[0] // feat_stride, self.input_shape[1] // feat_stride
        pool_outs = [feat]
        for b in bins:
            ph, pw = max(1, H // b), max(1, W // b)
            p = layers.AveragePooling2D(pool_size=(ph, pw),
                                        strides=(ph, pw),
                                        padding='same')(feat)
            p = self._conv_bn_relu(p, filters, kernel=1)
            p = ResizeLike(name=f'ppm_up_{b}')([p, feat])
            pool_outs.append(p)
        x = layers.Concatenate()(pool_outs)
        return self._conv_bn_relu(x, 512)

    def _vanilla_pspnet(self) -> Model:
        inputs = keras.Input(shape=self.input_shape, name='input')

        x    = self._conv_bn_relu(inputs, 64)
        x    = self._conv_bn_relu(x,  128, strides=2)
        x    = self._conv_bn_relu(x,  256, strides=2)
        feat = self._conv_bn_relu(x,  512, strides=2)   # stride 8

        x = self._ppm(feat, feat_stride=8)
        x = self._conv_bn_relu(x, 256)
        x = layers.UpSampling2D(size=8, interpolation='bilinear')(x)

        out = layers.Conv2D(1, 1, activation='sigmoid', name='output')(x)
        return Model(inputs, out, name='pspnet_vanilla')

    def _pspnet_with_backbone(self) -> Model:
        inputs = keras.Input(shape=self.input_shape, name='input')
        _, _, high = self._get_backbone_features(inputs)   # stride 16

        x = self._ppm(high, feat_stride=16)
        x = self._conv_bn_relu(x, 256)
        x = ResizeLike(name='upsample_to_input')([x, inputs])

        out = layers.Conv2D(1, 1, activation='sigmoid', name='output')(x)
        return Model(inputs, out, name=f'pspnet_{self.backbone_name}')

    # ------------------------------------------------------------------
    # DeepLabV3+ — ASPP
    # ------------------------------------------------------------------

    def _aspp(self, x, filters: int = 256):
        """Atrous Spatial Pyramid Pooling (Keras 3 compatível)."""
        p1 = self._conv_bn_relu(x, filters, kernel=1)

        p2 = layers.Conv2D(filters, 3, padding='same',
                           dilation_rate=6, activation='relu')(x)
        p2 = layers.BatchNormalization()(p2)

        p3 = layers.Conv2D(filters, 3, padding='same',
                           dilation_rate=12, activation='relu')(x)
        p3 = layers.BatchNormalization()(p3)

        p4 = layers.Conv2D(filters, 3, padding='same',
                           dilation_rate=18, activation='relu')(x)
        p4 = layers.BatchNormalization()(p4)

        # Image-level pooling — GlobalAveragePooling2D com keepdims evita tf.shape
        p5 = layers.GlobalAveragePooling2D(keepdims=True)(x)   # (B, 1, 1, C)
        p5 = self._conv_bn_relu(p5, filters, kernel=1)
        p5 = ResizeLike(name='aspp_img_pool_resize')([p5, x])

        out = layers.Concatenate()([p1, p2, p3, p4, p5])
        return self._conv_bn_relu(out, filters, kernel=1)

    def _deeplabv3plus_with_encoder(self, use_backbone: bool) -> Model:
        inputs = keras.Input(shape=self.input_shape, name='input')

        if use_backbone:
            low, _, high = self._get_backbone_features(inputs)
        else:
            x    = self._conv_bn_relu(inputs, 32, strides=2)
            low  = self._conv_bn_relu(x,  64)           # stride 2
            x    = self._conv_bn_relu(low, 128, strides=2)
            x    = self._conv_bn_relu(x,  256, strides=2)
            high = self._conv_bn_relu(x,  256)           # stride 8

        aspp_out = self._aspp(high)

        low_proj = self._conv_bn_relu(low, 48, kernel=1)

        # ASPP → mesmo tamanho de low_proj
        aspp_up = ResizeLike(name='aspp_upsample')([aspp_out, low_proj])

        x = layers.Concatenate()([aspp_up, low_proj])
        x = self._conv_bn_relu(x, 256)
        x = self._conv_bn_relu(x, 256)

        # upsample para resolução original
        x = ResizeLike(name='final_upsample')([x, inputs])

        out = layers.Conv2D(1, 1, activation='sigmoid', name='output')(x)
        name = (f'deeplabv3plus_{self.backbone_name}'
                if use_backbone else 'deeplabv3plus_vanilla')
        return Model(inputs, out, name=name)

    # ------------------------------------------------------------------
    # FPN — Feature Pyramid Network
    # ------------------------------------------------------------------

    def _vanilla_fpn(self) -> Model:
        """
        FPN vanilla (sem backbone externo).

        Bottom-up : 4 blocos de encoder produzem C2–C5 (strides 2/4/8/16).
        Top-down  : conexões laterais 1×1 + Add + refinamento 3×3 → P5→P2.
        Head      : P5–P2 upsampled para resolução de entrada → concat → sigmoid.
        """
        FPN_CH = 256
        inputs = keras.Input(shape=self.input_shape, name='input')

        # ── Bottom-up ────────────────────────────────────────────────
        def _enc(x, f):
            x = self._conv_bn_relu(x, f)
            x = self._conv_bn_relu(x, f)
            return x

        c2 = _enc(inputs, 64)
        c3 = _enc(layers.MaxPooling2D()(c2), 128)
        c4 = _enc(layers.MaxPooling2D()(c3), 256)
        c5 = _enc(layers.MaxPooling2D()(c4), 512)

        # ── Lateral projections ──────────────────────────────────────
        l5 = self._fpn_lateral(c5, FPN_CH)
        l4 = self._fpn_lateral(c4, FPN_CH)
        l3 = self._fpn_lateral(c3, FPN_CH)
        l2 = self._fpn_lateral(c2, FPN_CH)

        # ── Top-down fusion ──────────────────────────────────────────
        p5 = self._conv_bn_relu(l5, FPN_CH)
        p4 = self._conv_bn_relu(layers.Add()([ResizeLike()([p5, l4]), l4]), FPN_CH)
        p3 = self._conv_bn_relu(layers.Add()([ResizeLike()([p4, l3]), l3]), FPN_CH)
        p2 = self._conv_bn_relu(layers.Add()([ResizeLike()([p3, l2]), l2]), FPN_CH)

        # ── Segmentation head ────────────────────────────────────────
        up5 = ResizeLike(name='fpn_up_p5')([p5, inputs])
        up4 = ResizeLike(name='fpn_up_p4')([p4, inputs])
        up3 = ResizeLike(name='fpn_up_p3')([p3, inputs])
        up2 = ResizeLike(name='fpn_up_p2')([p2, inputs])

        x = layers.Concatenate()([up5, up4, up3, up2])
        x = self._conv_bn_relu(x, 256)
        x = layers.Dropout(0.3)(x)
        x = self._conv_bn_relu(x, 128)

        out = layers.Conv2D(1, 1, activation='sigmoid', name='output')(x)
        return Model(inputs, out, name='fpn_vanilla')

    def _fpn_with_backbone(self) -> Model:
        """
        FPN com backbone encoder.

        Usa as 3 escalas do backbone (low/medium/high) e adiciona um nível
        mais profundo via Conv stride=2 sobre 'high', totalizando 4 níveis:
          C_deep   ≈ stride 32
          C_high   ≈ stride 16
          C_medium ≈ stride  8
          C_low    ≈ stride  4

        Pirâmide top-down: lateral 1×1 + Add + refine 3×3.
        Head: todos os P upsampled para resolução de entrada → concat → sigmoid.
        """
        FPN_CH = 256
        inputs = keras.Input(shape=self.input_shape, name='input')
        low, medium, high = self._get_backbone_features(inputs)

        # ── Nível mais profundo (stride ~32) ─────────────────────────
        deep = layers.Conv2D(512, 3, strides=2, padding='same', use_bias=False)(high)
        deep = layers.BatchNormalization()(deep)
        deep = layers.Activation('relu')(deep)

        # ── Lateral projections ──────────────────────────────────────
        l_deep   = self._fpn_lateral(deep,   FPN_CH)
        l_high   = self._fpn_lateral(high,   FPN_CH)
        l_medium = self._fpn_lateral(medium, FPN_CH)
        l_low    = self._fpn_lateral(low,    FPN_CH)

        # ── Top-down fusion ──────────────────────────────────────────
        p_deep   = self._conv_bn_relu(l_deep, FPN_CH)
        p_high   = self._conv_bn_relu(
            layers.Add()([ResizeLike()([p_deep,   l_high]),   l_high]),   FPN_CH)
        p_medium = self._conv_bn_relu(
            layers.Add()([ResizeLike()([p_high,   l_medium]), l_medium]), FPN_CH)
        p_low    = self._conv_bn_relu(
            layers.Add()([ResizeLike()([p_medium, l_low]),    l_low]),    FPN_CH)

        # ── Segmentation head ────────────────────────────────────────
        up_deep   = ResizeLike(name='fpn_up_deep')  ([p_deep,   inputs])
        up_high   = ResizeLike(name='fpn_up_high')  ([p_high,   inputs])
        up_medium = ResizeLike(name='fpn_up_medium') ([p_medium, inputs])
        up_low    = ResizeLike(name='fpn_up_low')   ([p_low,    inputs])

        x = layers.Concatenate()([up_deep, up_high, up_medium, up_low])
        x = self._conv_bn_relu(x, 256)
        x = layers.Dropout(0.3)(x)
        x = self._conv_bn_relu(x, 128)

        out = layers.Conv2D(1, 1, activation='sigmoid', name='output')(x)
        return Model(inputs, out, name=f'fpn_{self.backbone_name}')


# ==============================================================================
# 5. EXEMPLO DE USO
# ==============================================================================

if __name__ == '__main__':
    # (head, backbone, shape, use_attention, loss)
    configs = [
        # U-Net vanilla 5 bandas — baseline
        ('unet', None,            (256, 256, 9),  False, 'bce_dice'),
        # U-Net com ResNet50 — 9 canais (dataset fotovoltaica)
        ('unet', 'resnet50',      (256, 256, 9),  False, 'bce_dice'),
        # Heavyweight Focus U-Net vanilla — 9 canais + HFL
        ('unet', None,            (256, 256, 9),  True,  'hybrid_focal'),
        # Heavyweight Focus U-Net + ResNet50 — 9 canais + HFL
        ('unet', 'resnet50',      (256, 256, 9),  True,  'hybrid_focal'),
        # Heavyweight Focus U-Net + EfficientNetB3
        ('unet', 'efficientnetb3',(256, 256, 9),  True,  'hybrid_focal'),
        # Heavyweight Focus U-Net + Xception
        ('unet', 'xception',      (256, 256, 9),  True,  'hybrid_focal'),
    ]

    for head, backbone, shape, attn, loss in configs:
        print(f'\n{"="*60}')
        label = 'Focus' if attn else 'Vanilla'
        print(f'{label} {head} | Backbone: {backbone} | Shape: {shape} | Loss: {loss}')
        factory = SegmentationModelFactory(head, backbone, shape,
                                           use_attention=attn, focal_gamma=1.25)
        model   = factory.build()
        factory.compile_model(loss=loss)
        print(f'Params: {model.count_params():,}')
        print(f'Output: {model.output_shape}')
