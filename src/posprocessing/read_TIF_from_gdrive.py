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
LOCAL_BASE_DIR = '/srv/almacen/db_images'

_KEY_4  = Path.home() / '.config/gcloud/keys/mapbiomas-caatinga-cloud04-78950c04489a.json'
_KEY_17 = Path.home() / '.config/gcloud/keys/ee-solkancengine17-ef2f5f6fe840.json'

SCOPES        = ['https://www.googleapis.com/auth/drive']
DRIVE_FOLDERS = ['dataset_fotovoltaica_TIFreg']

# ---------------------------------------------------------------------------
# Argumentos
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description='Baixa TIFs do Google Drive (pasta dataset_fotovoltaica_TIFreg)')
parser.add_argument('key_conta', type=str,
                    help='Conta de serviço a usar (ex: 4 → cloud04, qualquer outro → engine17)')
parser.add_argument('--dir-save-img', type=str, default=LOCAL_BASE_DIR,
                    help=f'Pasta local de destino (default: {LOCAL_BASE_DIR})')
parser.add_argument('--loop', action='store_true',
                    help='Fica em loop verificando a cada 1 hora (default: roda uma vez e sai)')
args = parser.parse_args()

unicaConta    = str(args.key_conta)
LOCAL_BASE_DIR = args.dir_save_img

SERVICE_ACCOUNT_FILE = str(_KEY_4 if unicaConta == '4' else _KEY_17)

# ---------------------------------------------------------------------------
# Autenticação
# ---------------------------------------------------------------------------
try:
    creds   = service_account.Credentials.from_service_account_file(
                  SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)

    about             = service.about().get(fields='user').execute()
    email_conectado   = about['user']['emailAddress']
    print('✅ Conectado com sucesso!')
    print(f'📧 Conta ativa: {email_conectado}')

except Exception as e:
    print(f'❌ Falha na conexão: {e}')
    sys.exit(1)


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_files_from_folder(folder_name: str, dest_base: str):
    print(f'\n--- Iniciando busca na pasta: {folder_name} ---')

    # Localiza pasta no Drive pelo nome
    query   = f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder'"
    results = service.files().list(q=query, fields='files(id, name)').execute()
    folders = results.get('files', [])

    if not folders:
        print(f'Aviso: Pasta {folder_name} não encontrada no Drive.')
        return

    folder_id = folders[0]['id']
    dest_path = os.path.join(dest_base, folder_name)
    os.makedirs(dest_path, exist_ok=True)

    files_baixados = 0
    next_page_token = None

    while True:
        file_results = service.files().list(
            q=f"'{folder_id}' in parents and trashed = false",
            fields='nextPageToken, files(id, name, size)',
            pageSize=100,
            pageToken=next_page_token,
        ).execute()
        files = file_results.get('files', [])

        if not files:
            print(f'Nenhum arquivo pendente em {folder_name}.')
            break

        for file in files:
            file_id   = file['id']
            file_name = file['name']
            file_path = os.path.join(dest_path, file_name)

            if os.path.exists(file_path):
                continue

            print(f'[{files_baixados + 1}] ⬇️  Baixando {file_name}...', end=' ', flush=True)

            try:
                request    = service.files().get_media(fileId=file_id)
                fh         = io.FileIO(file_path, 'wb')
                downloader = MediaIoBaseDownload(fh, request)

                done = False
                while not done:
                    _, done = downloader.next_chunk()
                fh.close()
                files_baixados += 1

                local_size = os.path.getsize(file_path)
                drive_size = int(file.get('size', 0))

                if local_size > 0 and (local_size == drive_size or drive_size == 0):
                    local_mb = local_size / (1024 * 1024)
                    print(f'✅ {local_mb:.2f} MB', end=' ')
                    # Tenta mover para lixeira — pode falhar se a conta de serviço
                    # não tiver permissão de escrita no arquivo (403), o que não
                    # invalida o download já concluído.
                    try:
                        service.files().update(
                            fileId=file_id, body={'trashed': True}
                        ).execute()
                        print('🗑️  movido para lixeira.')
                    except Exception as trash_err:
                        print(f'(sem permissão para mover lixeira: {trash_err})')
                else:
                    print(f'⚠️  Tamanho divergente (local {local_size} / drive {drive_size}). Mantendo.')

            except Exception as e:
                print(f'Erro ao baixar {file_name}: {e}')

        next_page_token = file_results.get('nextPageToken')
        if not next_page_token:
            break

    print(f'✅ Pasta {folder_name} concluída! {files_baixados} arquivos baixados.')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    os.makedirs(LOCAL_BASE_DIR, exist_ok=True)

    while True:
        print(f'\n[{time.strftime("%H:%M:%S")}] Iniciando download...')
        print(f'  Destino: {LOCAL_BASE_DIR}')

        for folder in DRIVE_FOLDERS:
            download_files_from_folder(folder, LOCAL_BASE_DIR)

        print('\n✅ Download concluído.')

        if not args.loop:
            break

        print('Aguardando 1 hora para a próxima verificação...')
        time.sleep(3600)