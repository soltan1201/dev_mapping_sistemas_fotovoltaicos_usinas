#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auditoria_pipeline.py
=====================
Audita as 880 combinações (região × ano) em todos os estágios do pipeline
fotovoltaico e, opcionalmente, dispara o script correto para cada lacuna.

Estágios verificados:
  1. NPY      → patches baixados  (<npy_dir>/<region>/<year>/patch_*.npy)
  2. PREDICT  → patches preditos  (<pred_dir>/<region>/<year>/patch_*_pred.npy)
  3. TIF      → GeoTIFF gerado    (<tif_dir>/pred_<region>_<year>.tif)
  4. GCS      → TIF no bucket     gs://<bucket>/<gcs_prefix>/pred_<region>_<year>.tif
  5. GEE      → asset no GEE      <asset_path>/reg_<region>_<year>_<model>_<backbone>

Uso:
  # Apenas relatório (sem executar nada)
  python auditoria_pipeline.py --config config_pipeline.json

  # Fixar estágio específico
  python auditoria_pipeline.py --config config_pipeline.json --fix tif
  python auditoria_pipeline.py --config config_pipeline.json --fix upload
  python auditoria_pipeline.py --config config_pipeline.json --fix transfer

  # Fixar tudo em cascata (tif → upload → transfer)
  python auditoria_pipeline.py --config config_pipeline.json --fix all

  # Filtrar
  python auditoria_pipeline.py --config config_pipeline.json --years 2022 2025 --regions 00000000000000000001

  # Salvar relatório CSV
  python auditoria_pipeline.py --config config_pipeline.json --report status.csv
"""

import argparse
import collections
import csv
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):  # fallback silencioso
        return it

import ee
from google.cloud import storage

collections.Callable = collections.abc.Callable

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────
ASSET_REGIONS = 'projects/mapbiomas-arida/update_02_05_2026_buffer_fotovoltaic_5km'
ALL_YEARS     = list(range(2016, 2026))   # 2016–2025 inclusive

MODELS = {
    'tif_fotovoltaicav1': 'unet_resnet50',
    'tif_fotovoltaicav2': 'unet_resnet101',
    'tif_fotovoltaicav3': 'unet_resnet152',
    'tif_fotovoltaicav4': 'unet_mobilenet',
    'tif_fotovoltaicav5': 'unet_resnext50',
    'tif_fotovoltaicav6': 'unet_xception',
}

_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROCESS_DIR = _SCRIPTS_DIR.parent / 'processClass'

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO PADRÃO (sobrescrita por --config)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    # Caminhos locais
    'npy_dir':      '/dados/dataset_fotovoltaica_npy',
    'predict_dir':  '/dados/predict_fotovoltaica',
    'tif_dir':      '~/db_images/tif_fotovoltaicav1',

    # GCS
    'bucket':       'mapbiomas-energia',
    'gcs_prefix':   'fotovoltaicas_tif/tif_fotovoltaicav1',
    'key_json':     '~/Dados/mapbiomas/mykeys/mapbiomas-agua-36521f541610.json',
    'gcp_project':  'mapbiomas-agua',

    # GEE
    'asset_path':   'projects/geo-data-s/assets/fotovoltaica/usinas_br_gc',
    'gee_project':  'geo-data-s',

    # Modelo
    'model_path':   'models/best_unet_resnet50_20260430_0257.keras',
    'input_format': 'npy',
    'batch_size':   8,

    # Cache local da lista de regiões (evita chamada GEE repetida)
    'regions_cache': '~/.cache/fv_regions.json',
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. REGIÕES DO GEE
# ─────────────────────────────────────────────────────────────────────────────

# def _safe_id(raw_id: str) -> str:
#     return str(raw_id).replace('/', '_').replace(':', '_')


def carregar_regioes(cache_path: Path, gee_project: str) -> list[str]:
    """
    Retorna lista de region_ids (sanitizados) do asset ASSET_REGIONS.
    Usa cache JSON se disponível; caso contrário consulta o GEE e salva.
    """
    cache_path = Path(cache_path).expanduser()
    if cache_path.exists():
        ids = json.loads(cache_path.read_text())
        log.info(f'Regiões carregadas do cache ({len(ids)}): {cache_path}')
        return ids

    log.info('Consultando GEE para listar regiões...')
    try:
        ee.Initialize(project=gee_project)
    except ee.EEException as exc:
        log.error(f'Falha ao inicializar GEE: {exc}')
        sys.exit(1)

    fc   = ee.FeatureCollection(ASSET_REGIONS)
    ids_list  = fc.reduceColumns(ee.Reducer.toList(), ['system:index']).get('list').getInfo()
    log.info(f'Regiões encontradas no GEE: {len(ids_list)}')

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(ids_list, indent=2, ensure_ascii=False))
    log.info(f'Cache salvo em: {cache_path}')
    return ids_list


# ─────────────────────────────────────────────────────────────────────────────
# 2. VERIFICAÇÕES POR ESTÁGIO
# ─────────────────────────────────────────────────────────────────────────────

def checar_npy(npy_dir: Path, region: str, year: int) -> bool:
    d = npy_dir / region / str(year)
    return d.is_dir() and any(d.glob('patch_*.npy'))


def checar_predict(pred_dir: Path, region: str, year: int) -> bool:
    d = pred_dir / region / str(year)
    return d.is_dir() and any(d.glob('patch_*_pred.npy'))


def checar_tif(tif_dir: Path, region: str, year: int) -> bool:
    return (tif_dir / f'pred_{region}_{year}.tif').exists()


def construir_set_gcs(bucket_name: str, prefix: str,
                      client: storage.Client) -> set[str]:
    """Retorna conjunto de 'region_year' presentes no bucket."""
    pat = re.compile(r'^pred_(.+)_(\d{4})\.tif$')
    s   = set()
    for b in client.list_blobs(bucket_name, prefix=prefix):
        m = pat.match(b.name.split('/')[-1])
        if m:
            s.add(f'{m.group(1)}_{m.group(2)}')
    return s


def construir_set_gee(asset_path: str) -> set[str]:
    """Retorna conjunto de asset names (só o basename) da ImageCollection, com paginação."""
    assets: set[str] = set()
    page_token = None
    while True:
        params = {'parent': asset_path}
        if page_token:
            params['pageToken'] = page_token
        response    = ee.data.listAssets(params)
        page_items  = response.get('assets', [])
        for item in page_items:
            assets.add(item['name'].split('/')[-1])
        log.info(f'  GEE listAssets: {len(assets)} assets carregados...')
        page_token = response.get('nextPageToken')
        if not page_token:
            break
    return assets


def construir_set_tasks_pendentes() -> set[str]:
    pendentes = set()
    try:
        for t in ee.data.listOperations():
            if t.get('metadata', {}).get('state', '') in ('RUNNING', 'PENDING'):
                desc = t.get('metadata', {}).get('description', '')
                if desc:
                    pendentes.add(desc)
    except Exception:
        pass
    return pendentes


# ─────────────────────────────────────────────────────────────────────────────
# 3. AUDITORIA MULTI-MODELO (TIF por backbone)
# ─────────────────────────────────────────────────────────────────────────────

def auditar_tifs_por_modelo(
    base_tif_dir: Path,
    regioes: list[str],
    years: list[int],
) -> dict[str, dict[str, list[int]]]:
    """
    Para cada backbone em MODELS, escaneia a pasta tif_fotovoltaicavX e
    retorna quais anos estão FALTANDO por região.

    Retorno:
        {
            'unet_resnet50': {
                '00000000000000000033': [2016, 2017, 2022],  # anos ausentes
                ...
            },
            ...
        }
    Regiões com todas as combinações presentes não aparecem no sub-dict.
    """
    base        = Path(base_tif_dir).expanduser()
    pat         = re.compile(r'^pred_(.+)_(\d{4})\.tif$')
    years_set   = set(years)
    regioes_set = set(regioes)

    resultado: dict[str, dict[str, list[int]]] = {}

    for folder_key, backbone in MODELS.items():
        tif_dir = base / folder_key
        presentes: dict[str, set[int]] = {r: set() for r in regioes}

        if tif_dir.is_dir():
            for f in tif_dir.iterdir():
                m = pat.match(f.name)
                if m:
                    region, year = m.group(1), int(m.group(2))
                    if region in regioes_set and year in years_set:
                        presentes[region].add(year)
        else:
            log.warning(f'[audit-models] Pasta não encontrada: {tif_dir}')

        faltando_por_regiao: dict[str, list[int]] = {}
        for region, anos_presentes in presentes.items():
            faltando = sorted(years_set - anos_presentes)
            if faltando:
                faltando_por_regiao[region] = faltando

        resultado[backbone] = faltando_por_regiao

    return resultado


def imprimir_audit_modelos(dict_registro: dict[str, dict[str, list[int]]]):
    log.info('=' * 75)
    log.info('AUDITORIA MULTI-MODELO — anos faltando por backbone / região')
    log.info('=' * 75)
    for backbone, regioes_dict in dict_registro.items():
        total_lacunas = sum(len(v) for v in regioes_dict.values())
        log.info(f'\n  [{backbone}]  regiões com lacuna: {len(regioes_dict)}'
                 f'  |  pares (região×ano) faltando: {total_lacunas}')
        for region, anos in sorted(regioes_dict.items()):
            log.info(f'    {region}  →  {anos}')
    log.info('=' * 75)


# ─────────────────────────────────────────────────────────────────────────────
# 4. RELATÓRIO
# ─────────────────────────────────────────────────────────────────────────────

COLS = ('npy', 'predict', 'tif', 'gcs', 'gee', 'gee_pending')

def proximo_estagio(row: dict) -> str:
    if not row['npy']:
        return 'download'
    if not row['predict']:
        return 'predict'
    if not row['tif']:
        return 'tif'
    if not row['gcs']:
        return 'upload'
    if not row['gee']:
        return 'transfer' if not row['gee_pending'] else 'aguardar_task'
    return '-'


def imprimir_relatorio(resultados: list[dict]):
    ok     = sum(1 for r in resultados if proximo_estagio(r) == '-')
    stages = collections.Counter(proximo_estagio(r) for r in resultados
                                  if proximo_estagio(r) != '-')

    log.info('=' * 75)
    log.info(f'TOTAL pares (região × ano) : {len(resultados)}')
    log.info(f'  ✓  Completos (GEE OK)    : {ok}')
    for stage, cnt in sorted(stages.items()):
        log.info(f'  ✗  Lacuna em [{stage:10s}] : {cnt}')
    log.info('=' * 75)

    # Tabela dos incompletos
    incompletos = [r for r in resultados if proximo_estagio(r) != '-']
    if incompletos:
        hdr = f"{'Região':<36} {'Ano':>4}  {'NPY':>3} {'PRD':>3} {'TIF':>3} {'GCS':>3} {'GEE':>3}  Próximo"
        log.info(hdr)
        log.info('-' * len(hdr))
        for r in incompletos[:200]:   # limita saída a 200 linhas
            def c(v): return '✓' if v else '✗'
            log.info(
                f"{r['region']:<36} {r['year']:>4}  "
                f"{c(r['npy']):>3} {c(r['predict']):>3} {c(r['tif']):>3} "
                f"{c(r['gcs']):>3} {c(r['gee']):>3}  {proximo_estagio(r)}"
            )
        if len(incompletos) > 200:
            log.info(f'  ... e mais {len(incompletos) - 200} linha(s) (veja o CSV)')


# ─────────────────────────────────────────────────────────────────────────────
# 5. EXECUÇÃO DOS SCRIPTS DE CORREÇÃO
# ─────────────────────────────────────────────────────────────────────────────

def _run(cmd: list[str], label: str):
    log.info(f'[FIX] {label}')
    log.info(f'  CMD: {" ".join(cmd)}')
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        log.error(f'  ERRO (código {ret.returncode})')
    else:
        log.info(f'  OK')


def fix_tif(faltando: list[dict], cfg: dict):
    """join_convert_npytoTIF para cada (region, year) sem TIF."""
    regioes = sorted({r['region'] for r in faltando})
    anos    = sorted({r['year']   for r in faltando})
    cmd = [
        sys.executable,
        str(_SCRIPTS_DIR / 'join_convert_npytoTIF.py'),
        '--predict-dir', cfg['predict_dir'],
        '--output-dir',  cfg['tif_dir'],
        '--regions', *regioes,
        '--years',   *map(str, anos),
    ]
    _run(cmd, f'TIF join — {len(faltando)} par(es)')


def fix_upload(faltando: list[dict], cfg: dict):
    """uploadTIF_from_localFolder_GoogleStorage para cada (region, year) sem GCS."""
    regioes = sorted({r['region'] for r in faltando})
    anos    = sorted({r['year']   for r in faltando})
    cmd = [
        sys.executable,
        str(_SCRIPTS_DIR / 'uploadTIF_from_localFolder_GoogleStorage.py'),
        '--tif-dir',    cfg['tif_dir'],
        '--bucket',     cfg['bucket'],
        '--gcs-prefix', cfg['gcs_prefix'],
        '--key-json',   cfg['key_json'],
        '--gcp-project', cfg['gcp_project'],
        '--regions', *regioes,
        '--years',   *map(str, anos),
    ]
    _run(cmd, f'Upload GCS — {len(faltando)} par(es)')


def fix_transfer(faltando: list[dict], cfg: dict, model: str, backbone: str):
    """transferTIF_fromGCSbucket_toGEEasset para cada (region, year) no GCS mas não no GEE."""
    regioes = sorted({r['region'] for r in faltando})
    anos    = sorted({r['year']   for r in faltando})
    cmd = [
        sys.executable,
        str(_SCRIPTS_DIR / 'transferTIF_fromGCSbucket_toGEEasset.py'),
        '--gcs-path',   cfg['gcs_prefix'],
        '--bucket',     cfg['bucket'],
        '--asset-path', cfg['asset_path'],
        '--project',    cfg['gee_project'],
        '--regions', *regioes,
        '--years',   *map(str, anos),
    ]
    _run(cmd, f'Transfer GEE — {len(faltando)} par(es)')


def fix_predict(faltando: list[dict], cfg: dict):
    """makePredict_patchinServer_v2 para pares com NPY mas sem predição."""
    regioes = sorted({r['region'] for r in faltando})
    anos    = sorted({r['year']   for r in faltando})
    cmd = [
        sys.executable,
        str(_PROCESS_DIR / 'makePredict_patchinServer_v2.py'),
        '--model-path',   cfg['model_path'],
        '--input-dir',    cfg['npy_dir'],
        '--input-format', cfg['input_format'],
        '--output-dir',   cfg['predict_dir'],
        '--batch-size',   str(cfg['batch_size']),
        '--regions', *regioes,
        '--years',   *map(str, anos),
    ]
    _run(cmd, f'Predict — {len(faltando)} par(es)')


def fix_download(faltando: list[dict], cfg: dict):
    """
    download_dataset_predict_tfrecord_fotovoltaica não suporta --regions.
    Executa um processo por ano para os anos com lacunas.
    O script interno pula regiões/patches já existentes (idempotente).
    """
    anos = sorted({r['year'] for r in faltando})
    for year in anos:
        cmd = [
            sys.executable,
            str(_SCRIPTS_DIR / 'download_dataset_predict_tfrecord_fotovoltaica.py'),
            '--output-dir',  cfg['npy_dir'],
            '--year_inic',   str(year),
            '--year_end',    str(year),
        ]
        _run(cmd, f'Download NPY — ano {year}')


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Audita e corrige lacunas no pipeline fotovoltaico')

    parser.add_argument('--config',   type=Path, default=None,
                        help='JSON de configuração (sobrescreve defaults)')
    parser.add_argument('--years',    type=int, nargs='+', default=None,
                        help='Filtrar anos: 2 valores = intervalo; 1+ = lista')
    parser.add_argument('--regions',  type=str, nargs='+', default=None,
                        help='Filtrar regiões (padrão: todas as 88)')
    parser.add_argument('--fix',      type=str, default=None,
                        choices=['download', 'predict', 'tif', 'upload', 'transfer', 'all'],
                        help='Estágio a corrigir (default: apenas relatório)')
    parser.add_argument('--report',   type=Path, default=None,
                        help='Salvar relatório CSV neste arquivo')
    parser.add_argument('--refresh-cache', action='store_true',
                        help='Ignorar cache e reler lista de regiões do GEE')
    parser.add_argument('--audit-models', action='store_true',
                        help='Audita TIFs de todos os backbones e exibe anos faltando')
    parser.add_argument('--audit-json',   type=Path, default=None,
                        help='Salvar resultado de --audit-models como JSON')
    parser.add_argument('--lacunas-gee-json', type=Path, default=None,
                        help='Salvar lacunas do GEE no formato {backbone:{region:[anos]}} '
                             '(compatível com --lacunas-json do transferTIF)')
    parser.add_argument('--all-models', action='store_true',
                        help='Audita todos os modelos de MODELS de uma vez e salva '
                             'lacunas_gee_v1..v6.json (requer --lacunas-gee-json como prefixo)')
    args = parser.parse_args()

    # ── Configuração ──────────────────────────────────────────────────────────
    cfg = DEFAULT_CONFIG.copy()
    if args.config:
        cfg.update(json.loads(Path(args.config).read_text()))

    # Expandir ~ em caminhos
    for key in ('npy_dir', 'predict_dir', 'tif_dir', 'key_json',
                'model_path', 'regions_cache'):
        cfg[key] = str(Path(cfg[key]).expanduser())

    tif_dir  = Path(cfg['tif_dir'])
    npy_dir  = Path(cfg['npy_dir'])
    pred_dir = Path(cfg['predict_dir'])

    # ── Anos ─────────────────────────────────────────────────────────────────
    if args.years:
        years = list(range(args.years[0], args.years[1] + 1)) if len(args.years) == 2 else args.years
    else:
        years = ALL_YEARS

    # ── Inicializar GEE (necessário para listar assets) ───────────────────────
    try:
        ee.Initialize(project=cfg['gee_project'])
        log.info(f"GEE inicializado (project={cfg['gee_project']})")
    except ee.EEException as exc:
        log.error(f'Falha ao inicializar GEE: {exc}')
        sys.exit(1)

    # ── Regiões ───────────────────────────────────────────────────────────────
    cache_path = Path(cfg['regions_cache']).expanduser()
    if args.refresh_cache and cache_path.exists():
        cache_path.unlink()

    todas_regioes = carregar_regioes(cache_path, cfg['gee_project'])

    if args.regions:
        regioes = [r for r in todas_regioes if r in set(args.regions)]
    else:
        regioes = todas_regioes

    log.info(f'Regiões a auditar : {len(regioes)}')
    log.info(f'Anos a auditar    : {years}')
    log.info(f'Total de pares    : {len(regioes) * len(years)}')
    log.info('=' * 75)

    # ── Auditoria multi-modelo ─────────────────────────────────────────────────
    if args.audit_models:
        base_tif_dir = Path(cfg['tif_dir']).expanduser().parent
        log.info(f'Base TIF dir para audit-models: {base_tif_dir}')
        dict_registro = auditar_tifs_por_modelo(base_tif_dir, regioes, years)
        imprimir_audit_modelos(dict_registro)
        if args.audit_json:
            import json as _json
            with open(args.audit_json, 'w', encoding='utf-8') as fj:
                _json.dump(dict_registro, fj, indent=2, ensure_ascii=False)
            log.info(f'JSON salvo: {args.audit_json.resolve()}')
        return

    # ── Grupo / modelo ────────────────────────────────────────────────────────
    group      = cfg['gcs_prefix'].rstrip('/').split('/')[-1]
    model_full = MODELS.get(group, 'unet_resnet50')
    model, backbone = (model_full.split('_', 1) + [''])[:2]

    # ── Pré-carregar sets GCS e GEE (uma única chamada cada) ─────────────────
    log.info('Carregando lista de blobs GCS...')
    gcs_client  = storage.Client.from_service_account_json(
        cfg['key_json'], project=cfg['gcp_project'])
    set_gcs     = construir_set_gcs(cfg['bucket'], cfg['gcs_prefix'], gcs_client)
    log.info(f'  TIFs no GCS: {len(set_gcs)}')

    log.info('Carregando lista de imagens assets GEE...')
    log.info(' assets GEE... ' + cfg['asset_path'])
    set_gee     = construir_set_gee(cfg['asset_path'])
    # os.exit()
    # set_pending = construir_set_tasks_pendentes()
    # |  Tasks pendentes: {len(set_pending)}
    log.info(f'  Assets no GEE: {len(set_gee)}  ')

    # ── Auditoria todos os modelos de uma vez ─────────────────────────────────
    if args.all_models:
        pares        = [(r, y) for r in regioes for y in years]
        lacunas_todos: dict[str, dict[str, list[int]]] = {}

        for folder_key, model_full_am in MODELS.items():
            m_am, b_am = (model_full_am.split('_', 1) + [''])[:2]
            faltando: dict[str, list[int]] = {}

            for region, year in tqdm(pares, desc=f'Auditando {model_full_am}', unit='par'):
                asset_short = f'reg_{region}_{year}_{m_am}_{b_am}'
                if asset_short not in set_gee:
                    faltando.setdefault(region, []).append(year)

            lacunas_todos[model_full_am] = faltando
            total_p = sum(len(v) for v in faltando.values())
            log.info(f'[{model_full_am}] regiões com lacuna: {len(faltando)}  '
                     f'pares faltando: {total_p}')

        if args.lacunas_gee_json:
            with open(args.lacunas_gee_json, 'w', encoding='utf-8') as fj:
                json.dump(lacunas_todos, fj, indent=2, ensure_ascii=False)
            log.info(f'JSON todos os modelos salvo: {args.lacunas_gee_json.resolve()}')
        return

    # ── Auditoria ─────────────────────────────────────────────────────────────
    pares = [(r, y) for r in regioes for y in years]   # 880 pares conhecidos
    resultados = []
    for region, year in tqdm(pares, desc='Auditando pares', unit='par'):
        gcs_key     = f'{region}_{year}'
        asset_short = f'reg_{region}_{year}_{model}_{backbone}'  # basename usado no GEE

        row = {
            'region':  region,
            'year':    year,
            'npy':     checar_npy(npy_dir, region, year),
            'predict': checar_predict(pred_dir, region, year),
            'tif':     checar_tif(tif_dir, region, year),
            'gcs':     gcs_key in set_gcs,
            'gee':     asset_short in set_gee,
        }
        resultados.append(row)

    imprimir_relatorio(resultados)

    # ── JSON lacunas GEE (formato compatível com transferTIF --lacunas-json) ────
    if args.lacunas_gee_json:
        faltando_gee: dict[str, list[int]] = {}
        for r in resultados:
            if not r['gee']:
                faltando_gee.setdefault(r['region'], []).append(r['year'])
        lacunas_dict = {f'{model}_{backbone}': faltando_gee}
        with open(args.lacunas_gee_json, 'w', encoding='utf-8') as fj:
            json.dump(lacunas_dict, fj, indent=2, ensure_ascii=False)
        total_pares = sum(len(v) for v in faltando_gee.values())
        log.info(f'JSON lacunas GEE salvo: {args.lacunas_gee_json.resolve()} '
                 f'({len(faltando_gee)} regiões, {total_pares} pares)')

    # ── Relatório CSV ─────────────────────────────────────────────────────────
    if args.report:
        fieldnames = ['region', 'year', 'npy', 'predict', 'tif', 'gcs', 'gee',
                      'gee_pending', 'proximo_estagio']
        with open(args.report, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in resultados:
                w.writerow({**r, 'proximo_estagio': proximo_estagio(r)})
        log.info(f'CSV salvo: {args.report.resolve()}')

    # ── Correções ─────────────────────────────────────────────────────────────
    if not args.fix:
        return

    fix_ops = ([args.fix] if args.fix != 'all'
               else ['download', 'predict', 'tif', 'upload', 'transfer'])

    for op in fix_ops:
        if op == 'download':
            falt = [r for r in resultados if not r['npy']]
            if falt:
                fix_download(falt, cfg)
            else:
                log.info('[download] Nenhuma lacuna encontrada.')

        elif op == 'predict':
            falt = [r for r in resultados if r['npy'] and not r['predict']]
            if falt:
                fix_predict(falt, cfg)
            else:
                log.info('[predict] Nenhuma lacuna encontrada.')

        elif op == 'tif':
            falt = [r for r in resultados if r['predict'] and not r['tif']]
            if falt:
                fix_tif(falt, cfg)
            else:
                log.info('[tif] Nenhuma lacuna encontrada.')

        elif op == 'upload':
            falt = [r for r in resultados if r['tif'] and not r['gcs']]
            if falt:
                fix_upload(falt, cfg)
            else:
                log.info('[upload] Nenhuma lacuna encontrada.')

        elif op == 'transfer':
            falt = [r for r in resultados if r['gcs'] and not r['gee']
                    and not r['gee_pending']]
            if falt:
                fix_transfer(falt, cfg, model, backbone)
            else:
                log.info('[transfer] Nenhuma lacuna encontrada.')


if __name__ == '__main__':
    main()
