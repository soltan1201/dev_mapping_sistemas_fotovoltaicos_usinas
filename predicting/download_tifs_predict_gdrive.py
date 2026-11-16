"""
Baixa todos os TIFs da pasta do Google Drive
    /content/drive/MyDrive/DS_FV_TIFs_scaled/PREDICT_V2
para a máquina local em
    ~/almacen/db_images/tif_planet_2025/

Baseado em src/posprocessing/read_TIF_from_gdrive.py.

A pasta precisa estar compartilhada com a conta de serviço usada.
Por padrão os arquivos NÃO são movidos para a lixeira do Drive (use --trash).

Uso:
    python predicting/download_tifs_predict_gdrive.py 4
    python predicting/download_tifs_predict_gdrive.py 17 --loop
    python predicting/download_tifs_predict_gdrive.py 4 --drive-path DS_FV_TIFs_scaled/PREDICT_V2 \
        --dir-save-img ~/almacen/db_images/tif_planet_2025
"""
import os
import io
import sys
import time
import argparse
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# ---------------------------------------------------------------------------
# Configurações padrão
# ---------------------------------------------------------------------------
LOCAL_BASE_DIR = str(Path.home() / 'almacen/db_images/tif_planet_2025')
DRIVE_PATH     = 'DS_FV_TIFs_scaled/PREDICT_V2'   # relativo a MyDrive

_KEY_4  = Path.home() / '.config/gcloud/keys/mapbiomas-caatinga-cloud04-78950c04489a.json'
_KEY_17 = Path.home() / '.config/gcloud/keys/ee-solkancengine17-ef2f5f6fe840.json'

SCOPES      = ['https://www.googleapis.com/auth/drive']
FOLDER_MIME = 'application/vnd.google-apps.folder'
TIF_EXTS    = ('.tif', '.tiff')
CHUNK_SIZE  = 50 * 1024 * 1024   # 50 MB por chunk

# ---------------------------------------------------------------------------
# Argumentos
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description=f'Baixa TIFs do Google Drive ({DRIVE_PATH}) para disco local')
parser.add_argument('key_conta', type=str,
                    help='Conta de serviço a usar (ex: 4 → cloud04, qualquer outro → engine17)')
parser.add_argument('--drive-path', type=str, default=DRIVE_PATH,
                    help=f'Caminho da pasta no Drive, relativo a MyDrive (default: {DRIVE_PATH})')
parser.add_argument('--dir-save-img', type=str, default=LOCAL_BASE_DIR,
                    help=f'Pasta local de destino (default: {LOCAL_BASE_DIR})')
parser.add_argument('--trash', action='store_true',
                    help='Move o arquivo para a lixeira do Drive após download verificado')
parser.add_argument('--loop', action='store_true',
                    help='Fica em loop verificando a cada 1 hora (default: roda uma vez e sai)')
args = parser.parse_args()

unicaConta     = str(args.key_conta)
LOCAL_BASE_DIR = os.path.expanduser(args.dir_save_img)

# aceita também o caminho estilo Colab: /content/drive/MyDrive/...
drive_path = args.drive_path
for prefix in ('/content/drive/MyDrive/', 'MyDrive/'):
    if drive_path.startswith(prefix):
        drive_path = drive_path[len(prefix):]
DRIVE_PARTS = [p for p in drive_path.strip('/').split('/') if p]

SERVICE_ACCOUNT_FILE = str(_KEY_4 if unicaConta == '4' else _KEY_17)

# ---------------------------------------------------------------------------
# Autenticação
# ---------------------------------------------------------------------------
try:
    creds   = service_account.Credentials.from_service_account_file(
                  SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)

    about           = service.about().get(fields='user').execute()
    email_conectado = about['user']['emailAddress']
    print('✅ Conectado com sucesso!')
    print(f'📧 Conta ativa: {email_conectado}')

except Exception as e:
    print(f'❌ Falha na conexão: {e}')
    sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers Drive
# ---------------------------------------------------------------------------

def _list(query: str, fields: str = 'files(id, name, mimeType, size)'):
    """Lista todos os itens que batem com a query (com paginação)."""
    items, token = [], None
    while True:
        res = service.files().list(
            q=query,
            fields=f'nextPageToken, {fields}',
            pageSize=1000,
            pageToken=token,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
        ).execute()
        items.extend(res.get('files', []))
        token = res.get('nextPageToken')
        if not token:
            return items


def resolve_folder_id(parts: list[str]) -> str | None:
    """Resolve 'A/B/C' → id da pasta C.

    A primeira pasta é buscada pelo nome (conta de serviço não enxerga o
    MyDrive do usuário, só o que foi compartilhado); as seguintes são buscadas
    como filhas da anterior.
    """
    parent_id = None
    for name in parts:
        safe = name.replace("'", "\\'")
        q = f"name = '{safe}' and mimeType = '{FOLDER_MIME}' and trashed = false"
        if parent_id:
            q += f" and '{parent_id}' in parents"
        found = _list(q, fields='files(id, name)')
        if not found:
            print(f"❌ Pasta '{name}' não encontrada"
                  + (f" dentro de {parent_id}" if parent_id else '')
                  + '. Verifique se está compartilhada com a conta de serviço.')
            return None
        if len(found) > 1:
            print(f"⚠️  {len(found)} pastas chamadas '{name}'; usando a primeira ({found[0]['id']}).")
        parent_id = found[0]['id']
    return parent_id


def iter_tifs(folder_id: str, rel: str = ''):
    """Percorre a pasta recursivamente devolvendo (arquivo, subcaminho relativo)."""
    for item in _list(f"'{folder_id}' in parents and trashed = false"):
        if item['mimeType'] == FOLDER_MIME:
            yield from iter_tifs(item['id'], os.path.join(rel, item['name']))
        elif item['name'].lower().endswith(TIF_EXTS):
            yield item, rel


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_file(file: dict, file_path: str) -> bool:
    """Baixa para <arquivo>.part e renomeia ao final. Retorna True se ok."""
    tmp_path = file_path + '.part'
    request  = service.files().get_media(fileId=file['id'], supportsAllDrives=True)
    with io.FileIO(tmp_path, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=CHUNK_SIZE)
        done = False
        while not done:
            status, done = downloader.next_chunk(num_retries=5)
            if status and not done:
                print(f'{int(status.progress() * 100)}%', end=' ', flush=True)

    local_size = os.path.getsize(tmp_path)
    drive_size = int(file.get('size', 0))
    if local_size > 0 and (drive_size == 0 or local_size == drive_size):
        os.replace(tmp_path, file_path)
        return True

    print(f'⚠️  Tamanho divergente (local {local_size} / drive {drive_size}).', end=' ')
    os.remove(tmp_path)
    return False


def download_folder(parts: list[str], dest_base: str):
    print(f"\n--- Iniciando busca na pasta: MyDrive/{'/'.join(parts)} ---")

    folder_id = resolve_folder_id(parts)
    if not folder_id:
        return

    files = list(iter_tifs(folder_id))
    print(f'📄 {len(files)} TIFs encontrados no Drive.')

    baixados = pulados = erros = 0
    for i, (file, rel) in enumerate(files, 1):
        dest_dir  = os.path.join(dest_base, rel)
        os.makedirs(dest_dir, exist_ok=True)
        file_path = os.path.join(dest_dir, file['name'])

        drive_size = int(file.get('size', 0))
        if os.path.exists(file_path) and (drive_size == 0 or os.path.getsize(file_path) == drive_size):
            pulados += 1
            continue

        print(f'[{i}/{len(files)}] ⬇️  {os.path.join(rel, file["name"])} '
              f'({drive_size / 1024**2:.1f} MB)...', end=' ', flush=True)
        try:
            if not download_file(file, file_path):
                erros += 1
                print()
                continue
            baixados += 1
            print('✅', end=' ')

            if args.trash:
                try:
                    service.files().update(
                        fileId=file['id'], body={'trashed': True},
                        supportsAllDrives=True,
                    ).execute()
                    print('🗑️  movido para lixeira.')
                except Exception as trash_err:
                    print(f'(sem permissão para mover lixeira: {trash_err})')
            else:
                print()

        except Exception as e:
            erros += 1
            print(f'❌ Erro: {e}')

    print(f'✅ Concluído: {baixados} baixados, {pulados} já existiam, {erros} erros.')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    os.makedirs(LOCAL_BASE_DIR, exist_ok=True)

    while True:
        print(f'\n[{time.strftime("%H:%M:%S")}] Iniciando download...')
        print(f'  Destino: {LOCAL_BASE_DIR}')

        download_folder(DRIVE_PARTS, LOCAL_BASE_DIR)

        if not args.loop:
            break

        print('Aguardando 1 hora para a próxima verificação...')
        time.sleep(3600)
