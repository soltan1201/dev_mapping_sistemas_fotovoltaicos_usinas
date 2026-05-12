#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
uploadTIF_from_localFolder_GoogleStorage.py
============================================
Faz upload dos GeoTIFFs gerados por join_convert_npytoTIF.py para um bucket
do Google Cloud Storage, convertendo para COG (Cloud Optimized GeoTIFF) via
gdal_translate antes do envio.

Estrutura esperada de entrada (pasta plana):
  <tif-dir>/pred_<region_id>_<year>.tif

Destino no bucket:
  <gcs-prefix>/<group>/pred_<region_id>_<year>.tif

  <group> = valor de --group, ou o nome da pasta --tif-dir se omitido.
  Exemplo: --tif-dir db_images/tif_fotovoltaicav1  →  group = tif_fotovoltaicav1
           --group grupo_v1                         →  group = grupo_v1

Uso:
  python uploadTIF_from_localFolder_GoogleStorage.py \\
      --tif-dir    ~/db_images/tif_fotovoltaicav1 \\
      --bucket     mapbiomas-energia \\
      --gcs-prefix fotovoltaicas_tif/tif_fotovoltaicav1 \\
      --key-json   ~/keys/mapbiomas-agua-36521f541610.json \\
      --years      2022 2025
"""

import argparse
import json
import logging
import os
import re
import tempfile
from pathlib import Path

from google.cloud import storage

# Mapeamento subpasta → backbone (mesmo de auditoria_pipeline.py)
MODELS = {
    'tif_fotovoltaicav1': 'unet_resnet50',
    'tif_fotovoltaicav2': 'unet_resnet101',
    'tif_fotovoltaicav3': 'unet_resnet152',
    'tif_fotovoltaicav4': 'unet_mobilenet',
    'tif_fotovoltaicav5': 'unet_resnext50',
    'tif_fotovoltaicav6': 'unet_xception',
}

LOG_FILE = Path('upload_tif_gcs.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
    ],
)
log = logging.getLogger(__name__)

# Padrão do nome gerado por join_convert_npytoTIF: pred_<region_id>_<year>.tif
_FNAME_RE = re.compile(r'^pred_(.+)_(\d{4})\.tif$')


def to_cog(src: Path, dst: Path):
    """Converte para Cloud Optimized GeoTIFF usando gdal_translate."""
    cmd = (
        f'gdal_translate {src} {dst} '
        f'-co TILED=YES -co COPY_SRC_OVERVIEWS=YES -co COMPRESS=LZW'
    )
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f'gdal_translate falhou (código {ret}): {src}')


def upload_file(src: Path, bucket_name: str, blob_name: str,
                key_json: str, project: str):
    client = storage.Client.from_service_account_json(key_json, project=project)
    bucket = client.bucket(bucket_name)
    blob   = bucket.blob(blob_name)
    blob.upload_from_filename(str(src))
    log.info(f'  upload OK → gs://{bucket_name}/{blob_name}')


def main():
    parser = argparse.ArgumentParser(
        description='Upload de GeoTIFFs (saída do join_convert_npytoTIF) para GCS')
    parser.add_argument('--tif-dir',    type=Path, required=True,
                        help='Pasta plana com os pred_*.tif')
    parser.add_argument('--bucket',     type=str,
                        default='mapbiomas-energia',
                        help='Nome do bucket GCS (padrão: mapbiomas-energia)')
    parser.add_argument('--gcs-prefix', type=str,
                        default='fotovoltaicas_tif',
                        help='Prefixo (pasta) dentro do bucket (padrão: fotovoltaicas_tif)')
    parser.add_argument('--key-json',   type=str,
                        default='~/Dados/projetos/mykeys/mapbiomas-agua-36521f541610.json',
                        help='Caminho para o JSON da service account GCP')
    parser.add_argument('--project',    type=str,
                        default='mapbiomas-agua',
                        help='Projeto GCP (padrão: mapbiomas-agua)')
    parser.add_argument('--years',      type=int, nargs='+', default=None,
                        help='Anos a enviar: 2 valores = intervalo inclusivo; '
                             '1 ou 3+ = lista explícita')
    parser.add_argument('--regions',    type=str, nargs='+', default=None,
                        help='IDs de região a enviar (padrão: todos)')
    parser.add_argument('--group',      type=str, default=None,
                        help='Subpasta dentro do gcs-prefix (padrão: nome da --tif-dir)')
    parser.add_argument('--tmp-dir',    type=Path, default=None,
                        help='Pasta para arquivos COG temporários (padrão: /tmp do sistema). '
                             'Use se /tmp estiver cheio.')
    parser.add_argument('--lacunas-json', type=Path, default=None,
                        help='JSON gerado pelo auditoria_pipeline --lacunas-gee-json; '
                             'envia apenas os pares (região × ano) listados como faltando')
    args = parser.parse_args()

    if args.years and len(args.years) == 2:
        args.years = list(range(args.years[0], args.years[1] + 1))

    group   = args.group or args.tif_dir.name
    backbone = MODELS.get(group, group)

    # Carrega pares faltando do JSON da auditoria (filtra somente o que falta no GEE)
    missing_pairs: set[tuple[str, int]] | None = None
    if args.lacunas_json:
        with open(args.lacunas_json, 'r', encoding='utf-8') as fj:
            lacunas = json.load(fj)
        backbone_data = lacunas.get(backbone, {})
        missing_pairs = {
            (region_id, ano)
            for region_id, anos in backbone_data.items()
            for ano in anos
        }
        log.info(f'Lacunas carregadas ({backbone}): {len(missing_pairs)} pares a enviar')

    tif_files = sorted(args.tif_dir.glob('pred_*.tif'))
    if not tif_files:
        log.warning(f'Nenhum arquivo pred_*.tif encontrado em {args.tif_dir}')
        return

    log.info('=' * 60)
    log.info(f'Fonte        : {args.tif_dir}  ({len(tif_files)} arquivos)')
    log.info(f'Bucket       : gs://{args.bucket}/{args.gcs_prefix}/{group}')
    log.info(f'Anos         : {args.years or "todos"}')
    log.info(f'Regiões      : {args.regions or "todas"}')
    log.info('=' * 60)

    total = 0
    tmp_kwargs = {'prefix': 'cog_tmp_', 'dir': args.tmp_dir} if args.tmp_dir else {'prefix': 'cog_tmp_'}
    with tempfile.TemporaryDirectory(**tmp_kwargs) as tmpdir:
        for tif_path in tif_files:
            m = _FNAME_RE.match(tif_path.name)
            if not m:
                log.warning(f'  nome fora do padrão, pulando: {tif_path.name}')
                continue

            region_id, year = m.group(1), int(m.group(2))

            if args.years   and year      not in args.years:
                continue
            if args.regions and region_id not in args.regions:
                continue
            if missing_pairs is not None and (region_id, year) not in missing_pairs:
                continue

            blob_name = f'{args.gcs_prefix}/{group}/{tif_path.name}'
            cog_path  = Path(tmpdir) / tif_path.name

            log.info(f'Processando  {tif_path.name}  (region={region_id}  year={year})')
            try:
                to_cog(tif_path, cog_path)
                upload_file(cog_path, args.bucket, blob_name, args.key_json, args.project)
                total += 1
            except Exception as exc:
                log.error(f'  Erro em {tif_path.name}: {exc}')
            finally:
                # libera espaço em disco imediatamente após upload
                if cog_path.exists():
                    cog_path.unlink()

    log.info(f'Concluído. Arquivos enviados: {total}  |  Log: {LOG_FILE.resolve()}')


if __name__ == '__main__':
    main()
