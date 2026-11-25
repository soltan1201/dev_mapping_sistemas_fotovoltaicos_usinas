#!/usr/bin/env python3
"""
Baixa TFRecords do Google Drive para o servidor I7 usando a credencial do gcloud
===============================================================================
Copia a pasta DS_FV_tfs_scaled_v3 (usada em train_fotovoltaica_colab_v2.ipynb)
do Drive para o servidor, onde já estão as amostras novas (tfrecord_fv_revisao),
para depois embaralhar tudo junto.

Usa a Drive API v3 com o token de `gcloud auth print-access-token`, sem rclone.
O gcloud pode ter várias contas logadas; o script usa a conta digitada
(a DONA da pasta), sem mexer na conta ativa nem no Earth Engine.

Pré-requisito (uma vez), no I7 sem navegador:

    gcloud auth login <conta_dona_da_pasta> --enable-gdrive-access --no-launch-browser
      → abra a URL num PC com navegador, logue nessa conta, cole o código.

    Conferir:  gcloud auth list
    Depois, se quiser, volte a conta ativa:  gcloud config set account <outra_conta>

- Mantém a estrutura de subpastas (train/ val/ test/).
- Arquivo que já existe no destino com o mesmo tamanho é pulado (rodar de novo
  retoma); grava em *.part e renomeia no fim. Nada é apagado no Drive.

Uso:
    python src/preprocessing/baixar_tfrecords_drive_gcloud.py
    Parâmetros pedidos via input(); Enter aceita o [padrão].
"""

import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# ==============================================================================
# 1. CONFIGURAÇÕES (padrões das perguntas)
# ==============================================================================
CONTA      = 'solkan.cengine17@gmail.com'
PASTA      = 'DS_FV_tfs_scaled_v3'
DESTINO    = '~/almacen/db_images/DS_FV_tfs_scaled_v3'
WORKERS    = 8
SUFIXO     = '.tfrecord.gz'
API        = 'https://www.googleapis.com/drive/v3'
FOLDER_MT  = 'application/vnd.google-apps.folder'
MAX_RETRY  = 5
CHUNK      = 8 * 1024 * 1024

# ==============================================================================
# 2. ENTRADA
# ==============================================================================

def ask(pergunta, padrao, cast=str):
    while True:
        txt = input(f'{pergunta} [{padrao}]: ').strip()
        if not txt:
            return padrao
        try:
            return cast(txt)
        except ValueError:
            print(f'  valor inválido: {txt}')


def ask_sim_nao(pergunta, padrao=False):
    txt = input(f'{pergunta} (s/n) [{"s" if padrao else "n"}]: ').strip().lower()
    return padrao if not txt else txt in ('s', 'sim', 'y', 'yes')

# ==============================================================================
# 3. TOKEN (gcloud) + DRIVE API
# ==============================================================================

class Token:
    """Access token do gcloud para uma conta; renovado a cada 45 min."""

    def __init__(self, conta):
        self.conta, self.valor, self.t0 = conta, None, 0
        self.lock = threading.Lock()

    def get(self, forcar=False):
        with self.lock:
            if forcar or self.valor is None or time.time() - self.t0 > 45 * 60:
                r = subprocess.run(['gcloud', 'auth', 'print-access-token',
                                    f'--account={self.conta}'],
                                   capture_output=True, text=True)
                if r.returncode != 0:
                    sys.exit(f'❌ gcloud não deu token para {self.conta}:\n{r.stderr.strip()}\n'
                             f'   Rode: gcloud auth login {self.conta} '
                             '--enable-gdrive-access --no-launch-browser')
                self.valor, self.t0 = r.stdout.strip(), time.time()
            return self.valor


def api_get(token, url, params=None, stream=False):
    for tentativa in range(MAX_RETRY):
        resp = requests.get(url, params=params, stream=stream, timeout=120,
                            headers={'Authorization': f'Bearer {token.get()}'})
        if resp.status_code == 401:
            token.get(forcar=True)
            continue
        if resp.status_code in (403, 429, 500, 502, 503) and tentativa < MAX_RETRY - 1:
            if resp.status_code == 403 and 'insufficient' in resp.text.lower():
                sys.exit('❌ Token sem acesso ao Drive. Refaça o login com '
                         '--enable-gdrive-access.')
            time.sleep(2 ** tentativa)
            continue
        resp.raise_for_status()
        return resp
    resp.raise_for_status()
    return resp


def listar(token, q):
    """Todos os arquivos que batem com a query q (com paginação)."""
    params = {'q': q, 'pageSize': 1000,
              'fields': 'nextPageToken, files(id, name, mimeType, size, parents)',
              'supportsAllDrives': 'true', 'includeItemsFromAllDrives': 'true'}
    files = []
    while True:
        data = api_get(token, f'{API}/files', params).json()
        files += data.get('files', [])
        if not data.get('nextPageToken'):
            return files
        params['pageToken'] = data['nextPageToken']


def achar_pasta(token, nome):
    nome_q = nome.replace("'", "\\'")
    pastas = listar(token, f"name = '{nome_q}' and mimeType = '{FOLDER_MT}' and trashed = false")
    if not pastas:
        sys.exit(f"❌ Pasta '{nome}' não encontrada nessa conta (nem compartilhada com ela).")
    if len(pastas) == 1:
        return pastas[0]['id']
    print(f"Há {len(pastas)} pastas chamadas '{nome}':")
    for k, p in enumerate(pastas, 1):
        n = len(listar(token, f"'{p['id']}' in parents and trashed = false"))
        print(f'  {k}) id={p["id"]}  ({n} itens)')
    k = ask('Qual usar (número)', 1, int)
    return pastas[k - 1]['id']


def listar_recursivo(token, pasta_id, rel=Path('.')):
    """Lista (rel_path, id, size) de todos os *.tfrecord.gz sob a pasta."""
    itens = []
    for f in listar(token, f"'{pasta_id}' in parents and trashed = false"):
        if f['mimeType'] == FOLDER_MT:
            itens += listar_recursivo(token, f['id'], rel / f['name'])
        elif f['name'].endswith(SUFIXO):
            itens.append((rel / f['name'], f['id'], int(f.get('size', 0))))
    return itens


def baixar(token, file_id, size, destino):
    if destino.exists() and destino.stat().st_size == size:
        return 'pulado'
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_name(destino.name + '.part')
    for tentativa in range(MAX_RETRY):
        try:
            with api_get(token, f'{API}/files/{file_id}',
                         {'alt': 'media', 'supportsAllDrives': 'true',
                          'acknowledgeAbuse': 'true'}, stream=True) as resp, \
                 open(tmp, 'wb') as fh:
                for bloco in resp.iter_content(CHUNK):
                    fh.write(bloco)
            if size and tmp.stat().st_size != size:
                raise IOError(f'tamanho {tmp.stat().st_size} != {size}')
            tmp.rename(destino)
            return 'baixado'
        except Exception as exc:
            if tentativa == MAX_RETRY - 1:
                tmp.unlink(missing_ok=True)
                return f'erro: {exc}'
            time.sleep(2 ** tentativa)

# ==============================================================================
# 4. PIPELINE
# ==============================================================================

def main():
    print('=' * 60)
    print('Parâmetros de entrada (Enter = padrão)')
    print('=' * 60)
    conta   = ask('Conta Google dona da pasta (logada no gcloud)', CONTA)
    pasta   = ask('Nome da pasta no Drive', PASTA)
    destino = Path(ask('Pasta de destino no servidor', DESTINO)).expanduser()
    workers = ask('Downloads em paralelo', WORKERS, int)
    dry_run = ask_sim_nao('Dry-run (só listar)?', False)

    token = Token(conta)
    token.get()
    print(f'\nToken OK para {conta}. Procurando pasta {pasta} ...')
    pasta_id = achar_pasta(token, pasta)

    print('Listando arquivos (recursivo)...', flush=True)
    itens = listar_recursivo(token, pasta_id)
    total_gb = sum(s for _, _, s in itens) / 1e9
    por_split = {}
    for rel, _, _ in itens:
        por_split[str(rel.parent)] = por_split.get(str(rel.parent), 0) + 1
    print(f'{len(itens)} arquivos *{SUFIXO}  ({total_gb:.2f} GB)')
    for split, n in sorted(por_split.items()):
        print(f'  {split}: {n}')

    pend = [i for i in itens
            if not ((destino / i[0]).exists() and (destino / i[0]).stat().st_size == i[2])]
    print(f'Já no destino: {len(itens) - len(pend)}  |  pendentes: {len(pend)}')

    if dry_run or not pend:
        return
    print(f'\nResumo: {conta}:{pasta} → {destino} | workers={workers}')
    if not ask_sim_nao('Confirmar e continuar?', True):
        return

    t0, ok, erros = time.time(), 0, []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(baixar, token, fid, size, destino / rel): rel
                for rel, fid, size in pend}
        for n, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            if res.startswith('erro'):
                erros.append((futs[fut], res))
                print(f'\n  ❌ {futs[fut]}: {res}')
            else:
                ok += 1
            print(f'\r  {n}/{len(pend)} arquivos  ({time.time() - t0:.0f}s)',
                  end='', flush=True)
    print(f'\n\nBaixados: {ok}  |  erros: {len(erros)}')
    if erros:
        print('Rode de novo para tentar só os que faltaram.')

    print('\nArquivos no destino por subpasta:')
    for split in sorted(por_split):
        n = len(list((destino / split).glob(f'*{SUFIXO}')))
        print(f'  {split}: {n}')
    print('Concluído.')


if __name__ == '__main__':
    main()
