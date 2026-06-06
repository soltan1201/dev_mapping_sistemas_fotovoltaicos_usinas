#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dashboard_predict.py
====================
Dashboard Dash para visualizar mosaicos NICFI e predições de segmentação.

Lê o predict_index.json gerado automaticamente por makePredict_fromTIF.py:
  {
    "mosaicos": "/abs/path/mosaicos_tif",
    "saidas": {
      "unet_efficientnetb7": "/abs/path/mosaicos_tif_unet_efficientnetb7",
      "unet_resnet152":      "/abs/path/mosaicos_tif_unet_resnet152"
    }
  }

Uso:
  python dashboard_predict.py \\
      --index-file ~/teste_dash/predict_index.json \\
      [--port 8050] [--max-display-px 1024]

Nomes esperados:
  mosaicos : <mosaicos_dir>/<region_id>_<year>.tif
  predições: <saida_dir>/pred_<region_id>_<year>.tif
"""

import argparse
import base64
import io
import json
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from scipy.ndimage import binary_dilation
from PIL import Image as PILImage

from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
_ap = argparse.ArgumentParser(description='Dashboard NICFI + predict fotovoltaico')
_ap.add_argument('--index-file',     type=Path, required=True,
                 help='predict_index.json gerado pelo makePredict_fromTIF.py')
_ap.add_argument('--port',           type=int,  default=8050)
_ap.add_argument('--max-display-px', type=int,  default=1024,
                 help='Dimensão máxima em pixels para display (padrão: 1024)')
_args = _ap.parse_args()

INDEX_FILE = _args.index_file.expanduser().resolve()
PORT       = _args.port
MAX_PX     = _args.max_display_px

# ---------------------------------------------------------------------------
# Carrega índice
# ---------------------------------------------------------------------------
if not INDEX_FILE.exists():
    raise FileNotFoundError(
        f'predict_index.json não encontrado: {INDEX_FILE}\n'
        f'Rode makePredict_fromTIF.py ao menos uma vez para gerá-lo.'
    )

_index      = json.loads(INDEX_FILE.read_text(encoding='utf-8'))
IMAGE_DIR   = Path(_index['mosaicos']).expanduser().resolve()
SAIDAS: dict = {
    k: Path(v).expanduser().resolve()
    for k, v in _index.get('saidas', {}).items()
}

log.info(f'Índice lido: {INDEX_FILE}')
log.info(f'Mosaicos   : {IMAGE_DIR}')
for k, v in SAIDAS.items():
    log.info(f'  {k}: {v}')

# ---------------------------------------------------------------------------
# Descoberta de arquivos
# ---------------------------------------------------------------------------

def _scan_images(folder: Path) -> dict:
    """Retorna {(region_id, year): Path} para <region>_<year>.tif"""
    result = {}
    for f in sorted(folder.glob('*.tif')):
        parts = f.stem.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            result[(parts[0], int(parts[1]))] = f
    return result


def _scan_predicts(folder: Path) -> dict:
    """Retorna {(region_id, year): Path} para pred_<region>_<year>.tif"""
    result = {}
    for f in sorted(folder.glob('pred_*.tif')):
        stem  = f.stem[5:]   # remove prefixo 'pred_'
        parts = stem.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            result[(parts[0], int(parts[1]))] = f
    return result


images_map = _scan_images(IMAGE_DIR)

# predicts_by_model: { model_key: {(region_id, year): Path} }
predicts_by_model: dict = {
    k: _scan_predicts(v) for k, v in SAIDAS.items()
}

all_model_keys = sorted(predicts_by_model.keys())

# Todas as chaves (region_id, year) que existem em qualquer lugar
all_keys = set(images_map.keys())
for pm in predicts_by_model.values():
    all_keys |= pm.keys()
all_keys = sorted(all_keys)

# region_id → anos disponíveis
region_years: dict = {}
for (rid, yr) in all_keys:
    region_years.setdefault(rid, []).append(yr)
for rid in region_years:
    region_years[rid] = sorted(region_years[rid])

all_regions = sorted(region_years.keys())

# Contagens por modelo
def _count_predicts(model_key: str) -> int:
    return len(predicts_by_model.get(model_key, {}))

def _count_pending(model_key: str) -> int:
    pm = predicts_by_model.get(model_key, {})
    return len(set(images_map.keys()) - pm.keys())

log.info(f'Mosaicos encontrados: {len(images_map)}')
for k in all_model_keys:
    log.info(f'  {k}: {_count_predicts(k)} predições | {_count_pending(k)} pendentes')

# ---------------------------------------------------------------------------
# Processamento de imagem
# ---------------------------------------------------------------------------

def _stretch_channel(arr: np.ndarray, p_low: int = 2, p_high: int = 98) -> np.ndarray:
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    lo = float(np.percentile(valid, p_low))
    hi = float(np.percentile(valid, p_high))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    return np.clip((arr - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


@lru_cache(maxsize=64)
def load_rgb(path_str: str, max_px: int) -> np.ndarray:
    """
    Lê o TIF de mosaico e retorna (H, W, 3) uint8 RGB com stretch p2-p98.
    Bandas do mosaico: 1=blue, 2=green, 3=red, 4=pvi, 5=pvpi (rasterio 1-indexed).
    """
    with rasterio.open(path_str) as src:
        H_full, W_full = src.height, src.width
        scale  = min(1.0, max_px / max(H_full, W_full))
        out_h  = max(1, int(H_full * scale))
        out_w  = max(1, int(W_full * scale))
        n_bands = src.count
        band_indices = [3, 2, 1] if n_bands >= 3 else [1, 1, 1]
        data = src.read(
            band_indices,
            out_shape=(3, out_h, out_w),
            resampling=Resampling.lanczos,
        )  # (3, H, W)

    rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    for c in range(3):
        rgb[:, :, c] = _stretch_channel(data[c].astype(np.float32))
    return rgb


@lru_cache(maxsize=128)
def load_mask(path_str: str, target_h: int, target_w: int) -> np.ndarray:
    """Lê predict TIF e retorna máscara binária bool (target_h, target_w)."""
    with rasterio.open(path_str) as src:
        data = src.read(
            1,
            out_shape=(target_h, target_w),
            resampling=Resampling.nearest,
        )
    return (data > 0.5) if data.dtype == np.float32 else data.astype(bool)


def make_boundary_overlay(rgb: np.ndarray, mask: np.ndarray,
                           color=(255, 230, 0), thickness: int = 3,
                           alpha: float = 0.92) -> np.ndarray:
    """Pinta a borda da mancha do predict em amarelo sobre o RGB."""
    dilated  = binary_dilation(mask, iterations=thickness)
    boundary = dilated & ~mask
    result   = rgb.copy()
    for c, val in enumerate(color):
        ch           = result[:, :, c].astype(np.float32)
        ch[boundary] = np.clip(val * alpha + ch[boundary] * (1.0 - alpha), 0, 255)
        result[:, :, c] = ch.astype(np.uint8)
    return result


def to_png_src(arr: np.ndarray) -> str:
    """Converte (H, W, 3) uint8 para data URI PNG base64."""
    buf = io.BytesIO()
    PILImage.fromarray(arr).save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()

# ---------------------------------------------------------------------------
# App Dash
# ---------------------------------------------------------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = 'NICFI · Predições Fotovoltaicas'

# Badges de estatísticas no topo (recalculados ao carregar)
def _build_stats_badges():
    badges = [
        dbc.Badge(f'{len(images_map)} mosaicos', color='secondary', className='me-2 fs-6'),
    ]
    for k in all_model_keys:
        n_pred = _count_predicts(k)
        n_pend = _count_pending(k)
        badges.append(
            dbc.Badge(
                f'{k}: {n_pred} predições  |  {n_pend} pendentes',
                color='success' if n_pend == 0 else 'warning',
                className='me-2 fs-6',
            )
        )
    return badges

_IMG_STYLE = {
    'width': '100%',
    'border': '1px solid #444',
    'borderRadius': '4px',
    'minHeight': '200px',
    'backgroundColor': '#111',
    'display': 'block',
}

_model_options = [{'label': k, 'value': k} for k in all_model_keys]

app.layout = dbc.Container([

    dbc.Row(dbc.Col(
        html.H4('NICFI · Predições Fotovoltaicas', className='text-white mt-3 mb-1')
    )),

    dbc.Row(dbc.Col(
        html.Div(_build_stats_badges(), className='mb-3')
    )),

    dbc.Row([
        dbc.Col([
            dbc.Label('Modelo', className='text-white-50 small'),
            dcc.Dropdown(
                id='dd-model',
                options=_model_options,
                value=all_model_keys[0] if all_model_keys else None,
                clearable=False,
                style={'color': '#000'},
            ),
        ], md=4),
        dbc.Col([
            dbc.Label('Região (system:index)', className='text-white-50 small'),
            dcc.Dropdown(
                id='dd-region',
                options=[{'label': r, 'value': r} for r in all_regions],
                value=all_regions[0] if all_regions else None,
                clearable=False,
                style={'color': '#000'},
            ),
        ], md=5),
        dbc.Col([
            dbc.Label('Ano', className='text-white-50 small'),
            dcc.Dropdown(id='dd-year', clearable=False, style={'color': '#000'}),
        ], md=3),
    ], className='mb-2'),

    dbc.Row(dbc.Col(html.Div(id='status-row', className='mb-2'))),

    dbc.Row([
        dbc.Col([
            html.P('RGB  ·  stretch p2–p98',
                   className='text-center text-muted small mb-1'),
            html.Img(id='img-rgb', style=_IMG_STYLE),
        ], md=6),
        dbc.Col([
            html.P('RGB  +  borda predict (amarelo)',
                   className='text-center text-muted small mb-1'),
            html.Img(id='img-overlay', style=_IMG_STYLE),
        ], md=6),
    ]),

], fluid=True, style={'backgroundColor': '#1a1a2e', 'minHeight': '100vh', 'padding': '0 24px'})


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output('dd-year', 'options'),
    Output('dd-year', 'value'),
    Input('dd-region', 'value'),
)
def cb_update_years(region_id):
    if not region_id:
        return [], None
    years = region_years.get(region_id, [])
    opts  = [{'label': str(y), 'value': y} for y in years]
    return opts, (years[-1] if years else None)   # default: ano mais recente


@app.callback(
    Output('img-rgb',    'src'),
    Output('img-overlay','src'),
    Output('status-row', 'children'),
    Input('dd-model',  'value'),
    Input('dd-region', 'value'),
    Input('dd-year',   'value'),
)
def cb_update_display(model_key, region_id, year):
    if not region_id or year is None or not model_key:
        return None, None, ''

    key      = (region_id, int(year))
    has_img  = key in images_map
    has_pred = key in predicts_by_model.get(model_key, {})

    badges = []
    if not has_img:
        badges.append(dbc.Badge('Mosaico não disponível', color='danger',  className='me-2'))
    if not has_pred:
        badges.append(dbc.Badge(f'Predição pendente ({model_key})', color='warning', className='me-2'))

    src_rgb = src_ov = None

    try:
        if has_img:
            rgb     = load_rgb(str(images_map[key]), MAX_PX)
            src_rgb = to_png_src(rgb)

            if has_pred:
                H, W   = rgb.shape[:2]
                pred_path = predicts_by_model[model_key][key]
                mask   = load_mask(str(pred_path), H, W)
                ov     = make_boundary_overlay(rgb, mask)
                src_ov = to_png_src(ov)
            else:
                src_ov = src_rgb   # sem predict: mostra RGB puro no painel direito

        elif has_pred:
            # orphan: predict existe mas sem mosaico original
            pred_path = predicts_by_model[model_key][key]
            mask  = load_mask(str(pred_path), MAX_PX, MAX_PX)
            gray  = mask.astype(np.uint8) * 255
            gray3 = np.stack([gray, gray, gray], axis=-1)
            src_ov = to_png_src(gray3)

    except Exception as exc:
        log.error(f'Erro ao carregar {region_id}/{year} ({model_key}): {exc}', exc_info=True)
        badges.append(dbc.Badge(f'Erro: {exc}', color='danger', className='me-2'))

    return src_rgb, src_ov, badges


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print(f'\n  Dashboard : http://localhost:{PORT}')
    print(f'  Índice    : {INDEX_FILE}')
    print(f'  Mosaicos  : {IMAGE_DIR}  ({len(images_map)} TIFs)')
    for k, v in SAIDAS.items():
        print(f'  {k}: {v}  ({_count_predicts(k)} predições)')
    print()
    app.run(host='0.0.0.0', port=PORT, debug=False)
