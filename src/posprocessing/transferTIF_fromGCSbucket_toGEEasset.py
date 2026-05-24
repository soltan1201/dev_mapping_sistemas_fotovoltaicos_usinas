#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transferTIF_fromGCSbucket_toGEEasset.py
========================================
Lê GeoTIFFs de um bucket GCS e os ingere como imagens em um
ImageCollection do Google Earth Engine (GEE).

Fluxo:
  1. Lista arquivos em gs://<bucket>/<gcs-path>/*.tif via gcloud storage ls
  2. Para cada arquivo: carrega com ee.Image.loadGeoTIFF e exporta para o asset

Uso:
  python transferTIF_fromGCSbucket_toGEEasset.py \\
      --gcs-path   fotovoltaicas_tif/tif_fotovoltaicav1 \\
      --bucket     mapbiomas-energia \\
      --asset-path projects/geo-data-s/assets/fotovoltaica/usinas_br \\
      --project    geo-data-s \\
      --version    1

  # criar pasta e coleção antes do primeiro uso:
  python transferTIF_fromGCSbucket_toGEEasset.py \\
      --gcs-path fotovoltaicas_tif/tif_fotovoltaicav1 \\
      --create-folder --create-collection
"""

import argparse
import json
import os
import subprocess
import sys
import collections

import ee

collections.Callable = collections.abc.Callable

# Mapeamento subpasta → modelo_backbone
MODELS = {
    "tif_fotovoltaicav1": "unet_resnet50",
    "tif_fotovoltaicav2": "unet_resnet101",
    "tif_fotovoltaicav3": "unet_resnet152",
    "tif_fotovoltaicav4": "unet_mobilenet",
    "tif_fotovoltaicav5": "unet_resnext50",
    "tif_fotovoltaicav6": "unet_xception",
    'tif_tfr_fotovoltaicav7': 'unet_efficientnetb7',
    'tif_tfr_fotovoltaicav8': 'unet_inceptionresnet',
}


def init_ee(project: str):
    try:
        ee.Initialize(project=project)
        print(f'Earth Engine inicializado (project={project})')
    except ee.EEException as e:
        print(f'Falha ao inicializar Earth Engine: {e}')
        sys.exit(1)


def export_to_asset(image: ee.Image, name: str, asset_path: str,
                    year: int, version: str, model: str, backbone: str, id_region: str, formato: str):
    asset_id = os.path.join(asset_path, name)
    data_inic = ee.Date.fromYMD(year, 12, 31)
    image = ee.Image(image).set(
        'year',               year,
        'version',            version,
        'backbone',           backbone,
        'modelo',             model,
        'region',             id_region, 
        'num_layer',          5,
        'formato',            formato,
        'semestre',           2,
        'system:time_start',  data_inic,
    ).selfMask()
    task = ee.batch.Export.image.toAsset(
        image=image,
        description=name,
        assetId=asset_id,
        scale=5,
        crs='EPSG:3857',
        region=image.geometry(),
        maxPixels=1e13,
        pyramidingPolicy={'.default': 'mode'},
    )
    task.start()
    print(f'  exportando → {asset_id}')


def main():
    parser = argparse.ArgumentParser(
        description='Transfere GeoTIFFs do GCS bucket para asset GEE')
    parser.add_argument('--gcs-path',   type=str, required=True,
                        help='Caminho no bucket: <prefix>/<group>  '
                             '(ex: fotovoltaicas_tif/tif_fotovoltaicav1)')
    parser.add_argument('--bucket',     type=str, default='mapbiomas-energia',
                        help='Nome do bucket GCS (padrão: mapbiomas-energia)')
    parser.add_argument('--asset-path', type=str,
                        default='projects/geo-data-s/assets/fotovoltaica/usinas_br',
                        help='ImageCollection de destino no GEE')
    parser.add_argument('--project',    type=str, default='geo-data-s',
                        help='Projeto GEE (padrão: geo-data-s)')
    parser.add_argument('--version',    type=str, default='1',
                        help='Versão a gravar como propriedade (padrão: 1)')
    parser.add_argument('--years',      type=int, nargs='+', default=None,
                        help='Filtrar anos: 2 valores = intervalo inclusivo; '
                             '1 ou 3+ = lista explícita')
    parser.add_argument('--regions',    type=str, nargs='+', default=None,
                        help='Filtrar regiões (padrão: todas)')
    parser.add_argument('--lacunas-json', type=str, default=None,
                        help='JSON gerado pelo auditoria_pipeline --lacunas-gee-json; '
                             'processa apenas os pares (região × ano) listados')
    parser.add_argument('--create-folder',     action='store_true',
                        help='Cria a pasta raiz no GEE antes de processar')
    parser.add_argument('--create-collection', action='store_true',
                        help='Cria a ImageCollection no GEE antes de processar')
    parser.add_argument('--formato', type=str, choices=['npy', 'tfr'], default='tfr',
                        help='Origem dos TIFs: npy (patches locais) ou tfr (GEE TFRecord). '
                             'Gravado como propriedade no asset GEE (padrão: tfr)')
    args = parser.parse_args()

    if args.years and len(args.years) == 2:
        args.years = list(range(args.years[0], args.years[1] + 1))

    init_ee(args.project)

    # Derivar modelo e backbone a partir do nome do grupo
    group = args.gcs_path.split('/')[-1]          # ex: tif_fotovoltaicav1
    model_full = MODELS.get(group, 'unet_resnet50')
    parts    = model_full.split('_', 1)
    model    = parts[0]                            # ex: unet
    backbone = parts[1] if len(parts) > 1 else ''  # ex: resnet50
    threhold = 0.5
    # Criar pasta / coleção no GEE se solicitado
    if args.create_folder:
        folder = '/'.join(args.asset_path.split('/')[:-1])
        cmd = f'earthengine create folder {folder}'
        os.system(cmd)
        print(f'Pasta criada: {folder}')

    if args.create_collection:
        cmd = f'earthengine create collection {args.asset_path}'
        os.system(cmd)
        print(f'ImageCollection criada: {args.asset_path}')

    # Listar arquivos no bucket
    gs_prefix = f'gs://{args.bucket}/{args.gcs_path}'
    cmd = f'gcloud storage ls "{gs_prefix}/*.tif"'
    print(f'Listando: {cmd}')
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f'Nenhum .tif encontrado em {gs_prefix}')
        print(f'  (gcloud stderr: {proc.stderr.strip()})')
        return
    tif_paths = [p for p in proc.stdout.strip().split('\n') if p.endswith('.tif')]

    if not tif_paths:
        print(f'Nenhum .tif encontrado em {gs_prefix}')
        return

    print(f'Encontrados {len(tif_paths)} arquivo(s). Iniciando ingestão...')
    print('=' * 60)

    # Carrega pares (região, ano) faltando do JSON da auditoria
    missing_pairs: set[tuple[str, int]] | None = None
    if args.lacunas_json:
        with open(args.lacunas_json, 'r', encoding='utf-8') as fj:
            lacunas = json.load(fj)
        backbone_data = lacunas.get(model_full, {})
        missing_pairs = {
            (region_id, ano)
            for region_id, anos in backbone_data.items()
            for ano in anos
        }
        print(f'Lacunas carregadas: {len(missing_pairs)} pares (região × ano) a processar')

    total = 0
    for cc, gcs_path in enumerate(tif_paths):
        name_tif  = gcs_path.split('/')[-1]          # pred_00000000000000000033_2022.tif
        nyear     = int(name_tif.split('_')[-1][:4]) # 2022
        id_region = str(name_tif.split('_')[1])
        region    = '_'.join(name_tif.split('_')[1:-1])

        if args.years   and nyear  not in args.years:
            continue
        if args.regions and region not in args.regions:
            continue

        # Se JSON fornecido, pula pares que NÃO estão nas lacunas (já existem no GEE)
        if missing_pairs is not None and (id_region, nyear) not in missing_pairs:
            print(f'#{cc:04d}  {name_tif}  [já no GEE — pulado]')
            continue

        base     = name_tif.replace('.tif', '').replace('pred', 'reg')
        namefile = f'{base}_{model}_{backbone}_{args.formato}'

        print(f'#{cc:04d}  {name_tif}  →  {namefile}')
        try:
            img = ee.Image.loadGeoTIFF(gcs_path)
            export_to_asset(img.gt(threhold), namefile, args.asset_path,
                            nyear, args.version, model,
                            backbone, id_region, args.formato
                    )
            total += 1
        except Exception as exc:
            print(f'  ERRO em {name_tif}: {exc}')

    print('=' * 60)
    print(f'Concluído. Tarefas submetidas: {total}')


if __name__ == '__main__':
    main()
