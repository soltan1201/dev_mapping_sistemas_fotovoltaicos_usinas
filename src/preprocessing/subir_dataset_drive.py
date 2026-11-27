#!/usr/bin/env python3
"""
Sobe o dataset de TFRecords (ex.: DS_FV_tfs_v4) para o Google Drive, sem rclone
================================================================================
Usa a Drive API v3 com o token de `gcloud auth print-access-token` da conta do
Colab. Não precisa de rclone, de chave JSON nem de client OAuth próprio.
Service account não serve: não tem cota no Meu Drive.

Pré-requisito (uma vez), no servidor:

    gcloud auth login <conta_do_colab> --enable-gdrive-access --no-launch-browser --force
      → abra a URL num navegador, logue NESSA conta, aceite o acesso ao Drive
        e cole o código.

O gcloud pode ter várias contas; o script usa a digitada, sem mudar a ativa
nem mexer no Earth Engine.

Etapas:
  1. mostra a conta conectada e o espaço livre no Drive;
  2. cria <pasta>/{train,val,test} no Meu Drive (ou reaproveita as existentes);
  3. upload resumable de cada arquivo; arquivo que já está lá com o mesmo
     tamanho é pulado (rodar de novo retoma); nada é apagado;
  4. confere nº de arquivos e tamanhos local × Drive por split;
  5. mostra as linhas para o notebook de treino.

Checklist: docs/specs/checklist_subir_dataset_drive.md

Uso:
    python src/preprocessing/subir_dataset_drive.py
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
CONTA     = 'solkan.cengine17@gmail.com'
LOCAL     = '~/almacen/db_images/DS_FV_tfs_v4'
PASTA     = 'DS_FV_tfs_v4'
WORKERS   = 4
SPLITS    = ['train', 'val', 'test']
EXTRAS    = ['manifesto.json']          # arquivos da raiz que também sobem

API       = 'https://www.googleapis.com/drive/v3'
UPLOAD    = 'https://www.googleapis.com/upload/drive/v3/files'
FOLDER_MT = 'application/vnd.google-apps.folder'
MAX_RETRY = 6

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


def gb(n):
    return f'{n / 1024**3:.2f} GB'

# ==============================================================================
# 3. TOKEN (gcloud) + DRIVE API
# ==============================================================================

class Token:
    """Access token do gcloud para uma conta; renovado a cada 40 min ou em 401."""

    def __init__(self, conta):
        self.conta, self.valor, self.t0 = conta, None, 0
        self.lock = threading.Lock()

    def get(self, forcar=False):
        with self.lock:
            if forcar or self.valor is None or time.time() - self.t0 > 40 * 60:
                r = subprocess.run(['gcloud', 'auth', 'print-access-token',
                                    f'--account={self.conta}'],
                                   capture_output=True, text=True, check=False)
                if r.returncode != 0:
                    sys.exit(f'❌ gcloud não deu token para {self.conta}:\n{r.stderr.strip()}\n'
                             f'   Rode: gcloud auth login {self.conta} --enable-gdrive-access '
                             '--no-launch-browser --force')
                self.valor, self.t0 = r.stdout.strip(), time.time()
            return self.valor

    def headers(self, extra=None):
        h = {'Authorization': f'Bearer {self.get()}'}
        h.update(extra or {})
        return h


def chamar(token, metodo, url, **kw):
    """Requisição com retry (401 renova token; 429/5xx espera e repete)."""
    extra = kw.pop('headers', None)
    for tentativa in range(MAX_RETRY):
        try:
            resp = requests.request(metodo, url, headers=token.headers(extra),
                                    timeout=kw.pop('timeout', 120), **kw)
        except requests.RequestException:
            if tentativa == MAX_RETRY - 1:
                raise
            time.sleep(2 ** tentativa)
            continue
        if resp.status_code == 401:
            token.get(forcar=True)
            continue
        if resp.status_code == 403 and 'insufficient' in resp.text.lower():
            sys.exit('❌ O token não tem acesso ao Drive. Refaça o login com '
                     '--enable-gdrive-access --force.')
        if resp.status_code in (403, 429, 500, 502, 503, 504) and tentativa < MAX_RETRY - 1:
            time.sleep(2 ** tentativa)
            continue
        resp.raise_for_status()
        return resp
    resp.raise_for_status()
    return resp


def listar(token, q, campos='files(id, name, mimeType, size)'):
    params = {'q': q, 'pageSize': 1000, 'fields': f'nextPageToken, {campos}'}
    files = []
    while True:
        data = chamar(token, 'GET', f'{API}/files', params=params).json()
        files += data.get('files', [])
        if not data.get('nextPageToken'):
            return files
        params['pageToken'] = data['nextPageToken']


def pasta_filha(token, nome, pai):
    """Id da pasta `nome` dentro de `pai` (cria se não existir)."""
    nome_q = nome.replace("'", "\\'")
    achadas = listar(token, f"name = '{nome_q}' and '{pai}' in parents and "
                            f"mimeType = '{FOLDER_MT}' and trashed = false")
    if achadas:
        if len(achadas) > 1:
            print(f"  ⚠️  {len(achadas)} pastas '{nome}'; usando a primeira.")
        return achadas[0]['id']
    meta = {'name': nome, 'mimeType': FOLDER_MT, 'parents': [pai]}
    return chamar(token, 'POST', f'{API}/files', json=meta,
                  params={'fields': 'id'}).json()['id']


def arquivos_na_pasta(token, pasta_id):
    """{nome: tamanho} dos arquivos (não pastas) dentro da pasta."""
    return {f['name']: int(f.get('size', 0))
            for f in listar(token, f"'{pasta_id}' in parents and trashed = false")
            if f['mimeType'] != FOLDER_MT}


def subir(token, caminho, pasta_id):
    """Upload resumable (sessão + PUT do arquivo). Retorna None ou mensagem de erro."""
    tam = caminho.stat().st_size
    for tentativa in range(MAX_RETRY):
        try:
            ini = chamar(token, 'POST', UPLOAD,
                         params={'uploadType': 'resumable', 'fields': 'id,size'},
                         json={'name': caminho.name, 'parents': [pasta_id]},
                         headers={'X-Upload-Content-Type': 'application/octet-stream',
                                  'X-Upload-Content-Length': str(tam)})
            sessao = ini.headers['Location']
            with open(caminho, 'rb') as fh:
                resp = requests.put(sessao, data=fh, timeout=1800,
                                    headers={'Content-Length': str(tam)})
            if resp.status_code in (200, 201):
                enviado = int(resp.json().get('size', tam))
                if enviado == tam:
                    return None
                raise OSError(f'tamanho no Drive {enviado} != local {tam}')
            raise OSError(f'HTTP {resp.status_code}: {resp.text[:200]}')
        except SystemExit:
            raise
        except Exception as exc:
            if tentativa == MAX_RETRY - 1:
                return str(exc)
            time.sleep(2 ** tentativa)

# ==============================================================================
# 4. PIPELINE
# ==============================================================================

def main():
    print('=' * 60)
    print('Parâmetros de entrada (Enter = padrão)')
    print('=' * 60)
    conta   = ask('Conta Google do Colab (logada no gcloud)', CONTA)
    local   = Path(ask('Pasta local do dataset', LOCAL)).expanduser()
    pasta   = ask('Pasta de destino no Drive (dentro do Meu Drive)', PASTA)
    workers = ask('Uploads em paralelo', WORKERS, int)
    dry_run = ask_sim_nao('Dry-run (só conferir, sem subir)?', False)

    # --- local ----------------------------------------------------------------
    if not local.is_dir():
        sys.exit(f'❌ Pasta local não encontrada: {local}')
    if (local / '_baldes_tmp').exists():
        sys.exit('❌ Existe _baldes_tmp/: o remontar_dataset_tfrecord.py não terminou.')
    arqs = {s: sorted((local / s).glob('*.tfrecord.gz')) for s in SPLITS}
    extras = [local / e for e in EXTRAS if (local / e).exists()]
    tam_total = sum(f.stat().st_size for fs in arqs.values() for f in fs)
    print(f'\nLocal: {local}  ({gb(tam_total)})')
    for s, fs in arqs.items():
        print(f'  {s}: {len(fs)} arquivos')
    if not sum(len(fs) for fs in arqs.values()):
        sys.exit('❌ Nenhum *.tfrecord.gz em train/ val/ test/.')

    # --- conta / espaço -------------------------------------------------------
    token = Token(conta)
    about = chamar(token, 'GET', f'{API}/about',
                   params={'fields': 'user(emailAddress), storageQuota'}).json()
    email = about['user']['emailAddress']
    quota = about.get('storageQuota', {})
    limite, usado = int(quota.get('limit', 0) or 0), int(quota.get('usage', 0))
    print(f'\nDrive conectado como: {email}')
    if email.lower() != conta.lower():
        print(f'⚠️  O token é de {email}, não de {conta}.')
    if limite:
        livre = limite - usado
        print(f'Espaço: usado {gb(usado)} de {gb(limite)}  |  livre {gb(livre)}')
        if livre < tam_total:
            print(f'⚠️  Espaço livre ({gb(livre)}) menor que o dataset ({gb(tam_total)}).'
                  ' Libere espaço (ex.: apagar DS_FV_tfs_scaled_v3 do Drive e esvaziar a'
                  ' lixeira) ou o upload vai parar com "storageQuotaExceeded".')
    else:
        print('Espaço: sem limite informado (conta com armazenamento ilimitado).')

    # --- pastas no Drive ------------------------------------------------------
    raiz_id = pasta_filha(token, pasta, 'root') if not dry_run else None
    ids, existentes = {}, {}
    if raiz_id:
        for s in SPLITS:
            ids[s] = pasta_filha(token, s, raiz_id)
            existentes[s] = arquivos_na_pasta(token, ids[s])
        existentes['_raiz'] = arquivos_na_pasta(token, raiz_id)

    tarefas = []
    for s, fs in arqs.items():
        for f in fs:
            if existentes.get(s, {}).get(f.name) != f.stat().st_size:
                tarefas.append((f, s))
    for f in extras:
        if existentes.get('_raiz', {}).get(f.name) != f.stat().st_size:
            tarefas.append((f, '_raiz'))
    tam_pend = sum(f.stat().st_size for f, _ in tarefas)
    print(f'\nPendentes: {len(tarefas)} arquivos ({gb(tam_pend)})'
          + ('' if raiz_id else '  [dry-run: não consultei o que já existe no Drive]'))
    if dry_run:
        return

    print(f'\nResumo: {local} → {email}:Meu Drive/{pasta} | workers={workers}')
    if not ask_sim_nao('Confirmar e continuar?', True):
        return

    # --- upload ---------------------------------------------------------------
    ids['_raiz'] = raiz_id
    t0, feito, erros = time.time(), 0, []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(subir, token, f, ids[s]): (f, s) for f, s in tarefas}
        for n, fut in enumerate(as_completed(futs), 1):
            f, s = futs[fut]
            err = fut.result()
            if err:
                erros.append((f, err))
                print(f'\n  ❌ {s}/{f.name}: {err}')
                if 'storageQuotaExceeded' in err:
                    print('  Drive cheio. Libere espaço e rode de novo.')
            else:
                feito += f.stat().st_size
            dt = time.time() - t0
            print(f'\r  {n}/{len(tarefas)} arquivos  {gb(feito)}/{gb(tam_pend)}  '
                  f'({feito / 1024**2 / max(dt, 1):.1f} MB/s, {dt / 60:.0f} min)',
                  end='', flush=True)
    print(f'\n\nEnviados: {len(tarefas) - len(erros)}  |  erros: {len(erros)}')

    # --- verificação ----------------------------------------------------------
    print('\nConferindo local × Drive...')
    ok = True
    print(f'  {"split":<6} {"local":>7} {"drive":>7}  tamanhos')
    for s, fs in arqs.items():
        no_drive = arquivos_na_pasta(token, ids[s])
        iguais = sum(1 for f in fs if no_drive.get(f.name) == f.stat().st_size)
        bate = iguais == len(fs)
        ok &= bate
        print(f'  {s:<6} {len(fs):>7} {len(no_drive):>7}  '
              f'{"✅" if bate else f"❌ {len(fs) - iguais} diferente(s)/faltando"}')

    if not ok or erros:
        print('\n⚠️  Upload incompleto. Rode o script de novo: ele só sobe o que falta.')
        return

    print('\n✅ Upload completo e conferido. No train_fotovoltaica_colab_v2.ipynb (célula 12):')
    print(f"   DATASET_ROOT_DS = '/content/drive/MyDrive/{pasta}'")
    print(f"   OUTPUT_DIR      = '/content/drive/MyDrive/TRAINING_OUTPUTS_FV_{pasta.split('_')[-1].upper()}'")
    print('   (recomendado) copiar para o disco do Colab antes do treino:')
    print(f"   !rsync -a --info=progress2 /content/drive/MyDrive/{pasta}/ /content/{pasta}/")
    print(f"   DATASET_ROOT_DS = '/content/{pasta}'")


if __name__ == '__main__':
    main()
