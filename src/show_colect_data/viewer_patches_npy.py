#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizador de Patches NPY — Fotovoltaica
==========================================
Interface Streamlit para navegar, visualizar e predizer patches .npy
com estrutura: <base_dir>/<region>/<year>/patch_*.npy

Executar:
  streamlit run viewer_patches_npy.py
"""

import sys
import math
import json
import numpy as np
import streamlit as st
from pathlib import Path

# ── opcionais ─────────────────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import tensorflow as tf
    import keras
    HAS_TF = True
except ImportError:
    HAS_TF = False

# ── constantes ────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve()

DEFAULT_BASE_DIR  = '/run/media/superuser/Almacen/imgDB/DS_FV_NICFI_TFRECORDS'
DEFAULT_MODEL_DIR = str(_HERE.parents[1] / 'models')
NORM_FACTOR       = 10_000.0

BAND_COMBOS = {
    'RGB natural (R-G-B)':  (2, 1, 0),
    'Falsa-cor NIR-R-G':    (3, 2, 1),
    'Falsa-cor NIR-G-B':    (3, 1, 0),
    'PVI-NIR-R':            (4, 3, 2),
    'EVI-NIR-R':            (7, 3, 2),
}

FEATURE_BANDS = ['blue', 'green', 'red', 'nir', 'pvi', 'iia', 'ri', 'evi']

# ── helpers ───────────────────────────────────────────────────────────────────

def arr_to_rgb(arr: np.ndarray, bands: tuple) -> np.ndarray:
    """(H,W,C) int16 → (H,W,3) uint8 com estiramento percentil 2–98."""
    channels = []
    for b in bands:
        ch = arr[:, :, b].astype(np.float32)
        p2, p98 = np.percentile(ch, 2), np.percentile(ch, 98)
        ch = (ch - p2) / (p98 - p2 + 1e-9)
        channels.append(np.clip(ch, 0, 1))
    return (np.stack(channels, axis=-1) * 255).astype(np.uint8)


@st.cache_data
def list_patches(year_dir_str: str) -> list:
    return sorted(str(p) for p in Path(year_dir_str).glob('patch_*.npy'))


@st.cache_data
def load_patch(npy_path_str: str) -> tuple:
    arr = np.load(npy_path_str)
    json_path = Path(npy_path_str).with_suffix('.json')
    meta = json.loads(json_path.read_text('utf-8')) if json_path.exists() else {}
    return arr, meta


@st.cache_resource
def load_model_cached(model_path_str: str):
    """Carrega modelo Keras com custom objects registrados com os packages corretos."""
    proc_dir = _HERE.parents[1] / 'processClass'
    sys.path.insert(0, str(proc_dir))
    from custom_losses import build_custom_objects
    return tf.keras.models.load_model(model_path_str, custom_objects=build_custom_objects())

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Viewer Patches NPY — Fotovoltaica',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ── sidebar: diretórios ───────────────────────────────────────────────────────
with st.sidebar:
    st.header('Diretórios')
    base_dir_str  = st.text_input('Base dos patches', DEFAULT_BASE_DIR)
    model_dir_str = st.text_input('Pasta dos modelos', DEFAULT_MODEL_DIR)

    base_dir  = Path(base_dir_str)
    model_dir = Path(model_dir_str)

    if not base_dir.exists():
        st.error(f'Diretório não encontrado:\n{base_dir}')
        st.stop()

    regions = sorted(d.name for d in base_dir.iterdir() if d.is_dir())
    if not regions:
        st.error('Nenhuma região encontrada.')
        st.stop()

    st.header('Navegação')
    region     = st.selectbox('Região', regions)
    region_dir = base_dir / region

    years_avail = sorted(d.name for d in region_dir.iterdir() if d.is_dir())
    year        = st.selectbox('Ano', years_avail)

    n_per_page = st.select_slider('Patches por grupo', [4, 6, 8, 10, 12], value=8)
    n_cols     = st.radio('Colunas', [2, 4], index=1, horizontal=True)

    combo_name = st.selectbox('Bandas para exibição', list(BAND_COMBOS.keys()))
    rgb_bands  = BAND_COMBOS[combo_name]

    st.header('Predição')
    model_files  = sorted(model_dir.glob('*.keras')) if model_dir.exists() else []
    model_names  = ['(nenhum)'] + [f.name for f in model_files]
    sel_model    = st.selectbox('Modelo', model_names)
    threshold    = st.slider('Threshold de binarização', 0.0, 1.0, 0.5, 0.05)

    if not HAS_TF:
        st.warning('TensorFlow não encontrado — predição desabilitada.')

# ── lista de patches ──────────────────────────────────────────────────────────
year_dir    = region_dir / year
patch_paths = list_patches(str(year_dir))
total       = len(patch_paths)

st.title(f'Patches NPY — {region} / {year}')

if total == 0:
    st.warning(f'Nenhum patch encontrado em `{year_dir}`')
    st.stop()

n_pages = math.ceil(total / n_per_page)

# ── session state: página e predições ─────────────────────────────────────────
nav_key = f'{region}|{year}'
if st.session_state.get('_nav_key') != nav_key:
    st.session_state['_nav_key']     = nav_key
    st.session_state['page']         = 0
    st.session_state['predictions']  = None
    st.session_state['pred_group']   = None

page = st.session_state.get('page', 0)
page = max(0, min(page, n_pages - 1))

# ── barra de navegação ────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns([1, 1, 5, 2])
with c1:
    if st.button('◀ Anterior', disabled=(page == 0)):
        st.session_state['page'] = page - 1
        st.session_state['predictions'] = None
        st.rerun()
with c2:
    if st.button('Próximo ▶', disabled=(page >= n_pages - 1)):
        st.session_state['page'] = page + 1
        st.session_state['predictions'] = None
        st.rerun()
with c3:
    start_disp = page * n_per_page + 1
    end_disp   = min((page + 1) * n_per_page, total)
    st.markdown(
        f'**Grupo {page + 1} / {n_pages}** &nbsp;|&nbsp; '
        f'patches {start_disp}–{end_disp} de {total}'
    )
with c4:
    target = st.number_input(
        'Ir para grupo', min_value=1, max_value=n_pages,
        value=page + 1, step=1, label_visibility='collapsed',
    )
    if target - 1 != page:
        st.session_state['page'] = target - 1
        st.session_state['predictions'] = None
        st.rerun()

page  = st.session_state['page']
start = page * n_per_page
end   = min(start + n_per_page, total)
group = patch_paths[start:end]

# ── carregar grupo atual ───────────────────────────────────────────────────────
patches = [load_patch(p) for p in group]

# ── botão de predição ─────────────────────────────────────────────────────────
can_predict  = (sel_model != '(nenhum)') and HAS_TF
predictions  = st.session_state.get('predictions')
pred_group   = st.session_state.get('pred_group')

if pred_group != (region, year, page):
    predictions = None

st.divider()
btn_col, clr_col, info_col = st.columns([2, 1, 4])

with btn_col:
    do_predict = st.button(
        f'Predizer {len(group)} patches',
        disabled=not can_predict,
        type='primary',
    )

with clr_col:
    if predictions is not None:
        if st.button('Limpar'):
            st.session_state['predictions'] = None
            st.rerun()

with info_col:
    if predictions is not None:
        n_pos = int((predictions > threshold).sum())
        n_tot = predictions.size
        st.info(
            f'Predição ativa — threshold {threshold:.2f} | '
            f'{n_pos:,} px positivos / {n_tot:,} px total '
            f'({100 * n_pos / n_tot:.1f}%)'
        )
    elif not can_predict and sel_model == '(nenhum)':
        st.caption('Selecione um modelo para habilitar a predição.')

if do_predict:
    model_path = model_dir / sel_model
    with st.spinner(f'Carregando modelo e rodando predição em {len(group)} patches…'):
        model = load_model_cached(str(model_path))
        batch = np.stack([
            arr.astype(np.float32) / NORM_FACTOR for arr, _ in patches
        ])
        preds       = model.predict(batch, verbose=0)   # (B, 256, 256, 1)
        predictions = preds[:, :, :, 0]
        st.session_state['predictions'] = predictions
        st.session_state['pred_group']  = (region, year, page)
    st.rerun()

# ── grid de imagens ───────────────────────────────────────────────────────────
st.divider()

for row_i in range(math.ceil(len(patches) / n_cols)):
    cols = st.columns(n_cols)
    for col_i in range(n_cols):
        idx = row_i * n_cols + col_i
        if idx >= len(patches):
            break

        arr, meta = patches[idx]
        r = meta.get('row', '?')
        c = meta.get('col', '?')
        lbl = f'r{r:04d}_c{c:04d}' if isinstance(r, int) else Path(group[idx]).stem

        with cols[col_i]:
            rgb = arr_to_rgb(arr, rgb_bands)

            if predictions is not None and HAS_MPL:
                pred = predictions[idx]

                fig, ax = plt.subplots(figsize=(3, 3))
                ax.imshow(rgb)
                ax.contour(pred, levels=[threshold],
                           colors='red', linewidths=1.2)
                ax.set_title(lbl, fontsize=7, pad=3)
                ax.axis('off')
                fig.tight_layout(pad=0.2)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            else:
                st.image(rgb, caption=lbl, use_container_width=True)

            with st.expander('Meta', expanded=False):
                st.json({
                    k: meta[k] for k in
                    ['row', 'col', 'year', 'origin_x', 'origin_y', 'crs']
                    if k in meta
                })
