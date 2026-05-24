#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verificar_e_limpar_bucket.py
============================
Verifica se cada TIF do GCS bucket foi ingerido com sucesso no GEE
e (opcionalmente) apaga do bucket os arquivos confirmados.

Fluxo:
  1. Lista TIFs em gs://<bucket>/<gcs-prefix>/
  2. Deriva o asset_id esperado para cada TIF (mesmo padrão do transferTIF_*)
  3. Consulta o GEE para confirmar existência do asset
  4. Exibe relatório: OK / FALTANDO / TASK PENDENTE
  5. Com --delete: apaga do GCS apenas os arquivos com asset confirmado (OK)

Nomenclatura esperada:
  GCS  : pred_<region_id>_<year>.tif
  Asset: reg_<region_id>_<year>_<model>_<backbone>

Uso:
    # Apenas relatório (sem apagar nada)
    python verificar_e_limpar_bucket.py \
        --gcs-prefix fotovoltaicas_tif/tif_fotovoltaicav1 \
        --bucket     mapbiomas-energia \
        --asset-path projects/geo-data-s/assets/fotovoltaica/usinas_br_gc \
        --project    geo-data-s

    # Confirmar e apagar do bucket
    python verificar_e_limpar_bucket.py \
        --gcs-prefix fotovoltaicas_tif/tif_fotovoltaicav1 \
        --bucket     mapbiomas-energia \
        --asset-path projects/geo-data-s/assets/fotovoltaica/usinas_br_gc \
        --project    geo-data-s \
        --delete

    # Salvar relatório CSV
    python verificar_e_limpar_bucket.py ... --report relatorio.csv
"""

import argparse
import csv
import logging
import re
import sys
from pathlib import Path

import ee
from google.cloud import storage

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ── Mapeamento grupo → modelo_backbone (igual ao transferTIF_*) ───────────────
MODELS = {
    'tif_fotovoltaicav1': 'unet_resnet50',
    'tif_fotovoltaicav2': 'unet_resnet101',
    'tif_fotovoltaicav3': 'unet_resnet152',
    'tif_fotovoltaicav4': 'unet_mobilenet',
    'tif_fotovoltaicav5': 'unet_resnext50',
    'tif_fotovoltaicav6': 'unet_xception',
    'tif_tfr_fotovoltaicav7': 'unet_efficientnetb7',
    'tif_tfr_fotovoltaicav8': 'unet_inceptionresnet',
    
}

_FNAME_RE = re.compile(r'^pred_(.+)_(\d{4})\.tif$')

STATUS_OK      = 'OK'
STATUS_MISSING = 'FALTANDO'
STATUS_PENDING = 'TASK_PENDENTE'
STATUS_ERROR   = 'ERRO_VERIFICACAO'


# ── GEE ──────────────────────────────────────────────────────────────────────

def init_ee(project: str):
    try:
        ee.Initialize(project=project)
        log.info(f'Earth Engine inicializado (project={project})')
    except ee.EEException as exc:
        log.error(f'Falha ao inicializar Earth Engine: {exc}')
        sys.exit(1)


def listar_assets_colecao(asset_path: str) -> set[str]:
    """Retorna conjunto de asset IDs (full name) presentes na ImageCollection."""
    assets = set()
    page_token = None
    while True:
        params = {'parent': asset_path}
        if page_token:
            params['pageToken'] = page_token
        resp = ee.data.listAssets(params)
        for item in resp.get('assets', []):
            assets.add(item['name'])
        page_token = resp.get('nextPageToken')
        if not page_token:
            break
    log.info(f'Assets encontrados na coleção: {len(assets)}')
    return assets


def listar_tasks_pendentes() -> set[str]:
    """Retorna conjunto de descriptions de tasks RUNNING ou PENDING."""
    pendentes = set()
    try:
        tasks = ee.data.listOperations()
        for t in tasks:
            state = t.get('metadata', {}).get('state', '')
            if state in ('RUNNING', 'PENDING'):
                desc = t.get('metadata', {}).get('description', '')
                if desc:
                    pendentes.add(desc)
    except Exception as exc:
        log.warning(f'Não foi possível listar tasks do GEE: {exc}')
    return pendentes


# ── GCS ──────────────────────────────────────────────────────────────────────

def listar_blobs(bucket_name: str, prefix: str,
                 key_json: str, project: str) -> list[str]:
    """Lista nomes de blobs .tif no prefixo dado."""
    client = storage.Client.from_service_account_json(key_json, project=project)
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    return [b.name for b in blobs if b.name.endswith('.tif')]


def apagar_blob(bucket_name: str, blob_name: str,
                key_json: str, project: str):
    client = storage.Client.from_service_account_json(key_json, project=project)
    client.bucket(bucket_name).blob(blob_name).delete()
    log.info(f'  APAGADO → gs://{bucket_name}/{blob_name}')


# ── Lógica principal ─────────────────────────────────────────────────────────

def derivar_asset_id(blob_name: str, asset_path: str,
                     model: str, backbone: str) -> str | None:
    """gs://bucket/prefix/pred_R_0001_2022.tif → asset_path/reg_R_0001_2022_unet_resnet50"""
    fname = blob_name.split('/')[-1]
    m = _FNAME_RE.match(fname)
    if not m:
        return None
    base     = fname.replace('.tif', '').replace('pred', 'reg')
    namefile = f'{base}_{model}_{backbone}'
    return f'{asset_path}/{namefile}'


def main():
    parser = argparse.ArgumentParser(
        description='Verifica assets GEE e (opcionalmente) apaga TIFs do GCS')

    parser.add_argument('--gcs-prefix',  required=True,
                        help='Prefixo no bucket, ex: fotovoltaicas_tif/tif_fotovoltaicav1')
    parser.add_argument('--bucket',      default='mapbiomas-energia',
                        help='Nome do bucket GCS (padrão: mapbiomas-energia)')
    parser.add_argument('--asset-path',
                        default='projects/geo-data-s/assets/fotovoltaica/usinas_br_gc',
                        help='ImageCollection destino no GEE')
    parser.add_argument('--project',     default='geo-data-s',
                        help='Projeto GEE (padrão: geo-data-s)')
    parser.add_argument('--key-json',
                        default='~/Dados/mapbiomas/mykeys/mapbiomas-agua-36521f541610.json',
                        help='Credencial service account GCP')
    parser.add_argument('--gcp-project', default='mapbiomas-agua',
                        help='Projeto GCP para autenticação GCS (padrão: mapbiomas-agua)')
    parser.add_argument('--years',       type=int, nargs='+', default=None,
                        help='Filtrar anos: 2 valores = intervalo; 1 ou 3+ = lista')
    parser.add_argument('--regions',     type=str, nargs='+', default=None,
                        help='Filtrar regiões (padrão: todas)')
    parser.add_argument('--delete',      action='store_true',
                        help='Apagar do GCS os arquivos com asset confirmado (OK)')
    parser.add_argument('--report',      type=Path, default=None,
                        help='Salvar relatório CSV neste arquivo')
    parser.add_argument('--all-models',  action='store_true',
                        help='Processa todas as subpastas de MODELS de uma vez '
                             '(ignora --gcs-prefix)')
    args = parser.parse_args()

    if args.years and len(args.years) == 2:
        args.years = list(range(args.years[0], args.years[1] + 1))

    key_json = str(Path(args.key_json).expanduser())

    # Monta lista de prefixos a processar
    if args.all_models:
        # ex: fotovoltaicas_tif/tif_fotovoltaicav1 → base = fotovoltaicas_tif
        base_prefix = args.gcs_prefix.rsplit('/', 1)[0]
        prefixos = [f'{base_prefix}/{group}' for group in MODELS]
    else:
        prefixos = [args.gcs_prefix]

    # Carrega assets GEE uma única vez (serve para todos os modelos)
    init_ee(args.project)
    log.info('Listando assets na coleção GEE...')
    assets_existentes = listar_assets_colecao(args.asset_path)
    log.info('Verificando tasks pendentes no GEE...')
    tasks_pendentes = listar_tasks_pendentes()

    todos_resultados = []
    for gcs_prefix in prefixos:
        _processar_prefixo(
            gcs_prefix, args, key_json, assets_existentes,
            tasks_pendentes, todos_resultados,
        )

    # Relatório CSV consolidado
    if args.report:
        with open(args.report, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['blob', 'asset_id', 'region', 'year', 'status'])
            writer.writeheader()
            writer.writerows(todos_resultados)
        log.info(f'Relatório salvo em: {args.report.resolve()}')


def _processar_prefixo(gcs_prefix: str, args, key_json: str,
                       assets_existentes: set, tasks_pendentes: set,
                       resultados: list):
    group      = gcs_prefix.rstrip('/').split('/')[-1]
    model_full = MODELS.get(group, 'unet_resnet50')
    model, backbone = (model_full.split('_', 1) + [''])[:2]

    log.info('=' * 65)
    log.info(f'Prefixo GCS  : gs://{args.bucket}/{gcs_prefix}')
    log.info(f'Modelo       : {model_full}')
    log.info(f'Modo         : {"APAGAR confirmados" if args.delete else "APENAS RELATÓRIO (dry-run)"}')
    log.info('=' * 65)

    blobs = listar_blobs(args.bucket, gcs_prefix, key_json, args.gcp_project)
    if not blobs:
        log.warning(f'Nenhum .tif encontrado em {gcs_prefix}')
        return

    log.info(f'TIFs encontrados no GCS: {len(blobs)}')

    contadores = {STATUS_OK: 0, STATUS_MISSING: 0,
                  STATUS_PENDING: 0, STATUS_ERROR: 0}
    marker = {STATUS_OK: '✓', STATUS_MISSING: '✗',
              STATUS_PENDING: '⏳', STATUS_ERROR: '?'}

    for blob_name in sorted(blobs):
        fname = blob_name.split('/')[-1]
        m = _FNAME_RE.match(fname)
        if not m:
            log.warning(f'  Nome fora do padrão, pulando: {fname}')
            continue

        region_id, year = m.group(1), int(m.group(2))

        if args.years   and year      not in args.years:
            continue
        if args.regions and region_id not in args.regions:
            continue

        asset_id = derivar_asset_id(blob_name, args.asset_path, model, backbone)
        if not asset_id:
            continue

        asset_short = asset_id.split('/')[-1]

        if asset_id in assets_existentes:
            status = STATUS_OK
        elif asset_short in tasks_pendentes:
            status = STATUS_PENDING
        else:
            status = STATUS_MISSING

        contadores[status] += 1
        resultados.append({
            'blob': blob_name, 'asset_id': asset_id,
            'region': region_id, 'year': year, 'status': status,
        })

        log.info(f'  {marker[status]}  {fname}  →  {status}')

        if status == STATUS_OK and args.delete:
            try:
                apagar_blob(args.bucket, blob_name, key_json, args.gcp_project)
            except Exception as exc:
                log.error(f'  Falha ao apagar {blob_name}: {exc}')

    total = sum(contadores.values())
    log.info(f'[{group}] OK:{contadores[STATUS_OK]}  '
             f'FALTANDO:{contadores[STATUS_MISSING]}  '
             f'PENDENTE:{contadores[STATUS_PENDING]}  '
             f'(total={total})')

    faltando = [r for r in resultados if r['status'] == STATUS_MISSING]
    if faltando:
        log.warning(f'  {len(faltando)} arquivo(s) SEM asset no GEE em {group}')


if __name__ == '__main__':
    main()
