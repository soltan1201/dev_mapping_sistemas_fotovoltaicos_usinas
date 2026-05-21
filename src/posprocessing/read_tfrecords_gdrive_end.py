#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
read_tfrecords_gdrive_end.py
============================
Baixa TFRecords da pasta do Google Drive para o servidor local e,
após confirmação de integridade, move os arquivos para a lixeira do Drive.

Pasta Drive : DS_FV_PREDICT_TFRECORDS  (conta caatinga04)
Destino local: ~/db_images/dataset_fotovoltaica_tf/

Uso:
  # Rodar uma vez
  python read_tfrecords_gdrive_end.py --key-json ~/.config/gcloud/keys/mapbiomas-caatinga-cloud04-78950c04489a.json

  # Loop contínuo (verifica a cada 1h)
  python read_tfrecords_gdrive_end.py --key-json ... --loop

  # Baixar sem apagar do Drive
  python read_tfrecords_gdrive_end.py --key-json ... --no-delete

  # Especificar pasta destino
  python read_tfrecords_gdrive_end.py --key-json ... --dest /caminho/para/destino

  # all argumentos 
 python read_tfrecords_gdrive_end.py \ 
   --key-json ~/.config/gcloud/keys/mapbiomas-caatinga-cloud04-78950c04489a.json \
   --folder DS_FV_PRED_TFRs \
   --dest /srv/almacen/db_images/dataset_fotovoltaica_tf_L5
  
"""

import argparse
import io
import logging
import sys
import time
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

SCOPES       = ['https://www.googleapis.com/auth/drive']
DRIVE_FOLDER = 'DS_FV_PRED_TFRs'
LOCAL_DEST   = Path('/srv/almacen/db_images/dataset_fotovoltaica_tf_L5').expanduser()


# ── Google Drive helpers ──────────────────────────────────────────────────────

def autenticar(key_json: str):
    creds   = service_account.Credentials.from_service_account_file(key_json, scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)
    about   = service.about().get(fields='user').execute()
    log.info(f'Conectado como: {about["user"]["emailAddress"]}')
    return service


def buscar_id_pasta(service, nome: str) -> str | None:
    query   = f"name = '{nome}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    results = service.files().list(q=query, fields='files(id, name)').execute()
    pastas  = results.get('files', [])
    if not pastas:
        log.warning(f'Pasta "{nome}" não encontrada no Drive.')
        return None
    log.info(f'Pasta encontrada: {nome}  (id={pastas[0]["id"]})')
    return pastas[0]['id']


def listar_arquivos_pasta(service, folder_id: str) -> list[dict]:
    """Lista todos os arquivos (com paginação) dentro de folder_id."""
    arquivos   = []
    page_token = None
    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields='nextPageToken, files(id, name, size)',
            pageSize=1000,
            pageToken=page_token,
        ).execute()
        arquivos  += resp.get('files', [])
        page_token = resp.get('nextPageToken')
        if not page_token:
            break
    return arquivos


def baixar_arquivo(service, file_id: str, dest: Path) -> bool:
    """Baixa um arquivo do Drive para dest. Retorna True se bem-sucedido."""
    try:
        request    = service.files().get_media(fileId=file_id)
        fh         = io.FileIO(str(dest), 'wb')
        downloader = MediaIoBaseDownload(fh, request, chunksize=8 * 1024 * 1024)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.close()
        return True
    except Exception as exc:
        log.error(f'  Erro ao baixar {dest.name}: {exc}')
        if dest.exists():
            dest.unlink()
        return False


def mover_para_lixeira(service, file_id: str, nome: str):
    """Move arquivo para a lixeira do Drive (não apaga permanentemente)."""
    try:
        service.files().update(fileId=file_id, body={'trashed': True}).execute()
        log.info(f'  🗑  Movido para lixeira: {nome}')
    except Exception as exc:
        log.warning(f'  Falha ao mover para lixeira {nome}: {exc}')


# ── Lógica principal ──────────────────────────────────────────────────────────

def processar_pasta(service, delete: bool, dest_dir: Path, folder_name: str = DRIVE_FOLDER):
    dest_dir.mkdir(parents=True, exist_ok=True)

    folder_id = buscar_id_pasta(service, folder_name)
    if not folder_id:
        return

    arquivos = listar_arquivos_pasta(service, folder_id)
    if not arquivos:
        log.info('Nenhum arquivo pendente na pasta do Drive.')
        return

    log.info(f'Arquivos encontrados no Drive: {len(arquivos)}')

    baixados = 0
    pulados  = 0
    erros    = 0

    for f in tqdm(arquivos, desc='Baixando TFRecords', unit='arquivo'):
        nome       = f['name']
        file_id    = f['id']
        drive_size = int(f.get('size', 0))
        dest       = dest_dir / nome

        if dest.exists():
            local_size = dest.stat().st_size
            if local_size > 0 and (local_size == drive_size or drive_size == 0):
                pulados += 1
                continue
            else:
                log.warning(f'  Arquivo incompleto localmente, re-baixando: {nome}')

        log.info(f'  ⬇  {nome}  ({drive_size / 1024 / 1024:.1f} MB)')
        ok = baixar_arquivo(service, file_id, dest)

        if not ok:
            erros += 1
            continue

        local_size = dest.stat().st_size
        if local_size > 0 and (local_size == drive_size or drive_size == 0):
            baixados += 1
            if delete:
                mover_para_lixeira(service, file_id, nome)
        else:
            log.warning(f'  Tamanho divergente — Local: {local_size}  Drive: {drive_size}. Mantendo no Drive.')
            erros += 1

    log.info('=' * 60)
    log.info(f'Baixados  : {baixados}')
    log.info(f'Pulados   : {pulados}  (já existiam)')
    log.info(f'Erros     : {erros}')
    log.info(f'Destino   : {dest_dir}')
    log.info('=' * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Baixa TFRecords do Google Drive (caatinga04) para o servidor')
    parser.add_argument(
        '--key-json', type=str,
        default='~/.config/gcloud/keys/mapbiomas-caatinga-cloud04-78950c04489a.json',
        help='Caminho para o JSON da service account (padrão: caatinga04)',
    )
    parser.add_argument(
        '--dest', type=Path,
        default=LOCAL_DEST,
        help=f'Pasta destino local (padrão: {LOCAL_DEST})',
    )
    parser.add_argument(
        '--folder', type=str,
        default=DRIVE_FOLDER,
        help=f'Nome da pasta no Google Drive (padrão: {DRIVE_FOLDER})',
    )
    parser.add_argument(
        '--no-delete', action='store_true',
        help='Não mover arquivos para a lixeira do Drive após download',
    )
    parser.add_argument(
        '--loop', action='store_true',
        help='Executar em loop contínuo (verifica a cada 1h)',
    )
    parser.add_argument(
        '--interval', type=int, default=3600,
        help='Intervalo do loop em segundos (padrão: 3600)',
    )
    args = parser.parse_args()

    key_json = str(Path(args.key_json).expanduser())
    if not Path(key_json).exists():
        log.error(f'Arquivo de credenciais não encontrado: {key_json}')
        sys.exit(1)

    dest_dir = Path(args.dest).expanduser()

    try:
        service = autenticar(key_json)
    except Exception as exc:
        log.error(f'Falha na autenticação: {exc}')
        sys.exit(1)

    delete = not args.no_delete

    if args.loop:
        while True:
            log.info(f'[{time.strftime("%H:%M:%S")}] Iniciando ciclo...')
            processar_pasta(service, delete, dest_dir, args.folder)
            log.info(f'Aguardando {args.interval}s para o próximo ciclo...')
            time.sleep(args.interval)
    else:
        processar_pasta(service, delete, dest_dir, args.folder)


if __name__ == '__main__':
    main()
