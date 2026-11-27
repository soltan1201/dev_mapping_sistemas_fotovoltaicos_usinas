#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dashboard_tfrecord.py
=====================
Dashboard Dash para inspecionar um dataset de TFRecords de treino
(ex.: DS_FV_tfs_v4 gerado por remontar_dataset_tfrecord.py).

  - Contagem de patches COM FV e SEM FV (label vazio) em train / val / test.
  - Amostra aleatória de patches do split escolhido (todos | com FV | sem FV):
    RGB (stretch p2–p98) e RGB + rótulo (preenchido vermelho, borda amarela).

As contagens leem todos os registros (só a banda label) em segundo plano;
o progresso aparece no topo. O resultado fica em <root>/.dash_stats_cache.json
e só arquivos novos ou alterados são relidos na próxima vez.

Formato esperado (o mesmo do train_fotovoltaica_colab_v2.ipynb):
  <root>/{train,val,test}/*.tfrecord.gz  (GZIP)
  bandas blue, green, red, pvi, pvpi + label, float_list 257x257
  (recortado para 256x256, como no treino).

Uso:
  python src/dash/dashboard_tfrecord.py \\
      --root ~/almacen/db_images/DS_FV_tfs_v4 [--port 8051] [--n-imgs 8]
"""

import argparse
import base64
import io
import json
import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from scipy.ndimage import binary_dilation

import tensorflow as tf
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger(__name__)
tf.get_logger().setLevel('ERROR')

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
_ap = argparse.ArgumentParser(description='Dashboard de inspeção de TFRecords FV')
_ap.add_argument('--root', type=Path, default=Path('~/almacen/db_images/DS_FV_tfs_v4'),
                 help='Pasta com subpastas train/ val/ test/')
_ap.add_argument('--port', type=int, default=8051)
_ap.add_argument('--n-imgs', type=int, default=8, help='Patches por sorteio (padrão: 8)')
_ap.add_argument('--workers', type=int, default=4, help='Threads da contagem (padrão: 4)')
_args = _ap.parse_args()

ROOT    = _args.root.expanduser().resolve()
PORT    = _args.port
N_IMGS  = _args.n_imgs
WORKERS = _args.workers

SPLITS     = ['train', 'val', 'test']
BANDS      = ['blue', 'green', 'red', 'pvi', 'pvpi']
LABEL      = 'label'
RAW        = 257
PATCH      = 256
CACHE_FILE = ROOT / '.dash_stats_cache.json'

FEAT_ALL   = {k: tf.io.FixedLenFeature([RAW, RAW], tf.float32) for k in BANDS + [LABEL]}
FEAT_LABEL = {LABEL: tf.io.FixedLenFeature([RAW, RAW], tf.float32)}

# ---------------------------------------------------------------------------
# Arquivos
# ---------------------------------------------------------------------------
if not ROOT.exists():
    raise FileNotFoundError(f'Pasta não encontrada: {ROOT}')

FILES = {s: sorted((ROOT / s).glob('*.tfrecord.gz')) for s in SPLITS if (ROOT / s).is_dir()}
if not FILES:
    raise FileNotFoundError(f'Nenhuma subpasta train/ val/ test/ em {ROOT}')

# ---------------------------------------------------------------------------
# Contagem em segundo plano (com cache)
# ---------------------------------------------------------------------------
_lock  = threading.Lock()
_stats = {}          # str(path) → {'size','mtime','n','fv_idx':[...],'fv_px'}
_prog  = {'feitos': 0, 'total': 0, 'rodando': False, 'erros': 0}


def _load_cache():
    try:
        return json.loads(CACHE_FILE.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _save_cache():
    with _lock:
        data = dict(_stats)
    try:
        tmp = CACHE_FILE.with_suffix('.tmp')
        tmp.write_text(json.dumps(data), encoding='utf-8')
        tmp.replace(CACHE_FILE)
    except Exception as exc:
        log.warning(f'Não foi possível gravar o cache: {exc}')


def _label_px(raw):
    lab = tf.io.parse_single_example(raw, FEAT_LABEL)[LABEL][:PATCH, :PATCH]
    lab = tf.where(tf.math.is_nan(lab), tf.zeros_like(lab), lab)
    return tf.reduce_sum(tf.cast(lab > 0, tf.int32))


def _stats_file(path):
    ds = (tf.data.TFRecordDataset(str(path), compression_type='GZIP')
          .map(_label_px, num_parallel_calls=tf.data.AUTOTUNE))
    px = [int(v) for v in ds.as_numpy_iterator()]
    st = path.stat()
    return {'size': st.st_size, 'mtime': st.st_mtime, 'n': len(px),
            'fv_idx': [i for i, v in enumerate(px) if v > 0], 'fv_px': sum(px)}


def _worker_contagem():
    cache = _load_cache()
    pend = []
    for files in FILES.values():
        for f in files:
            c = cache.get(str(f))
            st = f.stat()
            if c and c.get('size') == st.st_size and c.get('mtime') == st.st_mtime:
                _stats[str(f)] = c
            else:
                pend.append(f)
    _prog.update(total=len(pend), feitos=0, rodando=bool(pend))
    log.info(f'Contagem: {len(pend)} arquivos a ler ({len(_stats)} do cache)')
    if not pend:
        return
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(_stats_file, f): f for f in pend}
        for n, fut in enumerate(as_completed(futs), 1):
            f = futs[fut]
            try:
                res = fut.result()
                with _lock:
                    _stats[str(f)] = res
            except Exception as exc:
                _prog['erros'] += 1
                log.error(f'{f.name}: {exc}')
            _prog['feitos'] = n
            if n % 50 == 0:
                _save_cache()
    _prog['rodando'] = False
    _save_cache()
    log.info('Contagem concluída.')


def resumo():
    """{split: {'arquivos','lidos','n','fv','zero','fv_px'}}"""
    out = {}
    with _lock:
        for s, files in FILES.items():
            r = {'arquivos': len(files), 'lidos': 0, 'n': 0, 'fv': 0, 'fv_px': 0}
            for f in files:
                c = _stats.get(str(f))
                if c:
                    r['lidos'] += 1
                    r['n'] += c['n']
                    r['fv'] += len(c['fv_idx'])
                    r['fv_px'] += c['fv_px']
            r['zero'] = r['n'] - r['fv']
            out[s] = r
    return out

# ---------------------------------------------------------------------------
# Leitura de patches
# ---------------------------------------------------------------------------

def read_patch(path, k):
    raw = next(iter(tf.data.TFRecordDataset(str(path), compression_type='GZIP')
                    .skip(k).take(1)))
    ex = tf.io.parse_single_example(raw, FEAT_ALL)
    return {b: np.nan_to_num(ex[b].numpy()[:PATCH, :PATCH]) for b in BANDS + [LABEL]}


def sortear(split, filtro, n, seed=None):
    """Lista de (path, idx). Usa o cache quando disponível para filtrar."""
    rng   = random.Random(seed)
    files = FILES.get(split, [])
    cands = []
    with _lock:
        for f in files:
            c = _stats.get(str(f))
            if not c:
                continue
            fv = set(c['fv_idx'])
            if filtro == 'fv':
                idx = c['fv_idx']
            elif filtro == 'zero':
                idx = [i for i in range(c['n']) if i not in fv]
            else:
                idx = range(c['n'])
            cands.extend((f, i) for i in idx)
    if cands:
        return rng.sample(cands, min(n, len(cands)))
    if filtro != 'todos' or not files:
        return []
    # cache ainda vazio: escolhe arquivos e posições baixas (existem em qualquer shard)
    return [(rng.choice(files), rng.randint(0, 9)) for _ in range(n)]

# ---------------------------------------------------------------------------
# Imagem
# ---------------------------------------------------------------------------

def _stretch(arr, p_low=2, p_high=98):
    lo, hi = np.percentile(arr, [p_low, p_high])
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)
    return np.clip((arr - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def patch_rgb(p):
    return np.stack([_stretch(p['red']), _stretch(p['green']), _stretch(p['blue'])], axis=-1)


def overlay_label(rgb, mask, fill=(255, 40, 40), borda=(255, 230, 0), alpha=0.35):
    out = rgb.astype(np.float32)
    for c, v in enumerate(fill):
        out[..., c][mask] = out[..., c][mask] * (1 - alpha) + v * alpha
    contorno = binary_dilation(mask, iterations=1) & ~mask
    for c, v in enumerate(borda):
        out[..., c][contorno] = v
    return out.astype(np.uint8)


def to_png_src(arr, escala=2):
    img = PILImage.fromarray(arr)
    if escala > 1:
        img = img.resize((arr.shape[1] * escala, arr.shape[0] * escala), PILImage.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = 'TFRecords · Dataset FV'

_IMG = {'width': '100%', 'border': '1px solid #444', 'borderRadius': '4px',
        'backgroundColor': '#111', 'imageRendering': 'pixelated'}

app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H4(f'Dataset de treino FV · {ROOT.name}',
                            className='text-white mt-3 mb-1'))),
    dbc.Row(dbc.Col(html.Div(id='progresso', className='text-white-50 small mb-2'))),

    dbc.Row([
        dbc.Col(html.Div(id='tabela'), md=5),
        dbc.Col(dcc.Graph(id='grafico', config={'displayModeBar': False},
                          style={'height': '260px'}), md=7),
    ], className='mb-3'),

    dbc.Row([
        dbc.Col([
            dbc.Label('Dataset', className='text-white-50 small'),
            dbc.RadioItems(id='rd-split', inline=True,
                           options=[{'label': s, 'value': s} for s in FILES],
                           value=next(iter(FILES))),
        ], md=4),
        dbc.Col([
            dbc.Label('Patches', className='text-white-50 small'),
            dbc.RadioItems(id='rd-filtro', inline=True, value='todos',
                           options=[{'label': 'todos', 'value': 'todos'},
                                    {'label': 'com FV', 'value': 'fv'},
                                    {'label': 'sem FV', 'value': 'zero'}]),
        ], md=4),
        dbc.Col(dbc.Button('Sortear outros', id='bt-sortear', color='primary',
                           className='mt-3'), md=4),
    ], className='mb-2'),

    dbc.Row(dbc.Col(html.Div(id='status-sorteio', className='text-white-50 small mb-2'))),
    dbc.Row(id='grade'),

    dcc.Interval(id='tick', interval=3000, n_intervals=0),
], fluid=True, style={'backgroundColor': '#1a1a2e', 'minHeight': '100vh', 'padding': '0 24px'})


@app.callback(
    Output('progresso', 'children'),
    Output('tabela', 'children'),
    Output('grafico', 'figure'),
    Output('tick', 'disabled'),
    Input('tick', 'n_intervals'),
)
def cb_contagem(_):
    r = resumo()
    if _prog['rodando']:
        prog = (f'Contando patches... {_prog["feitos"]}/{_prog["total"]} arquivos lidos '
                f'(os números abaixo ainda são parciais)')
    else:
        prog = 'Contagem completa' + (f' · {_prog["erros"]} arquivo(s) com erro' if _prog['erros'] else '')

    linhas = [html.Tr([html.Th(h) for h in
                       ['split', 'arquivos', 'patches', 'com FV', 'sem FV', '% FV']])]
    tot = {'arquivos': 0, 'n': 0, 'fv': 0, 'zero': 0}
    for s, v in r.items():
        pct = 100 * v['fv'] / v['n'] if v['n'] else 0
        linhas.append(html.Tr([html.Td(s), html.Td(f'{v["lidos"]}/{v["arquivos"]}'),
                               html.Td(f'{v["n"]:,}'), html.Td(f'{v["fv"]:,}'),
                               html.Td(f'{v["zero"]:,}'), html.Td(f'{pct:.1f}%')]))
        for k in tot:
            tot[k] += v[k] if k != 'arquivos' else v['arquivos']
    pct = 100 * tot['fv'] / tot['n'] if tot['n'] else 0
    linhas.append(html.Tr([html.Th('total'), html.Th(tot['arquivos']), html.Th(f'{tot["n"]:,}'),
                           html.Th(f'{tot["fv"]:,}'), html.Th(f'{tot["zero"]:,}'),
                           html.Th(f'{pct:.1f}%')]))
    tabela = dbc.Table(linhas, bordered=False, dark=True, size='sm', className='mb-0')

    splits = list(r)
    fig = go.Figure([
        go.Bar(name='com FV', x=splits, y=[r[s]['fv'] for s in splits], marker_color='#f2b134'),
        go.Bar(name='sem FV', x=splits, y=[r[s]['zero'] for s in splits], marker_color='#5b6b8c'),
    ])
    fig.update_layout(barmode='group', template='plotly_dark', margin=dict(l=40, r=10, t=10, b=30),
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      legend=dict(orientation='h', y=1.12))
    return prog, tabela, fig, not _prog['rodando']


@app.callback(
    Output('grade', 'children'),
    Output('status-sorteio', 'children'),
    Input('rd-split', 'value'),
    Input('rd-filtro', 'value'),
    Input('bt-sortear', 'n_clicks'),
)
def cb_sorteio(split, filtro, _):
    escolhidos = sortear(split, filtro, N_IMGS)
    if not escolhidos:
        msg = ('Aguardando a contagem para filtrar por FV...' if _prog['rodando']
               else 'Nenhum patch encontrado com esse filtro.')
        return [], msg

    cards = []
    for path, k in escolhidos:
        try:
            p    = read_patch(path, k)
            rgb  = patch_rgb(p)
            mask = p[LABEL] > 0
            n_px = int(mask.sum())
            corpo = dbc.Row([
                dbc.Col(html.Img(src=to_png_src(rgb), style=_IMG), width=6),
                dbc.Col(html.Img(src=to_png_src(overlay_label(rgb, mask)), style=_IMG), width=6),
            ], className='g-1')
            rodape = html.Div([
                dbc.Badge(f'FV {n_px} px' if n_px else 'sem FV',
                          color='warning' if n_px else 'secondary', className='me-1'),
                html.Span(f'{path.name} #{k}', className='text-white-50',
                          style={'fontSize': '0.7rem', 'wordBreak': 'break-all'}),
            ], className='mt-1')
        except Exception as exc:
            log.error(f'{path.name}#{k}: {exc}')
            corpo, rodape = html.Div(f'Erro: {exc}', className='text-danger small'), None
        cards.append(dbc.Col(dbc.Card(dbc.CardBody([corpo, rodape], className='p-2'),
                                      color='dark', className='mb-3'), md=6, lg=3))

    msg = (f'{len(escolhidos)} patches de {split} ({filtro})  ·  '
           'esquerda: RGB p2–p98  ·  direita: rótulo (vermelho, borda amarela)')
    return cards, msg


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    threading.Thread(target=_worker_contagem, daemon=True).start()
    print(f'\n  Dashboard : http://localhost:{PORT}')
    print(f'  Dataset   : {ROOT}')
    for s, fs in FILES.items():
        print(f'  {s}: {len(fs)} arquivos')
    print()
    app.run(host='0.0.0.0', port=PORT, debug=False)
