#!/usr/bin/env python3
"""
Sobe o dataset de TFRecords (ex.: DS_FV_tfs_v4) do servidor para o Google Drive
================================================================================
Usa rclone com um remote logado como USUÁRIO na conta do Colab.
Service account não serve para subir: não tem cota no Meu Drive
("Service Accounts do not have storage quota").

Etapas:
  1. confere rclone, remote (login de usuário) e a pasta local;
  2. compara o tamanho local com o espaço livre no Drive;
  3. rclone copy (retoma de onde parou se rodar de novo; nada é apagado);
  4. rclone check --one-way + contagem por split no Drive;
  5. mostra as linhas para o notebook de treino.

Checklist completo: docs/specs/checklist_subir_dataset_drive.md

Uso:
    python src/preprocessing/subir_dataset_drive_rclone.py
    Parâmetros pedidos via input(); Enter aceita o [padrão].
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

# ==============================================================================
# 1. CONFIGURAÇÕES (padrões das perguntas)
# ==============================================================================
REMOTE    = 'ee-solkaneengine17'
LOCAL     = '~/almacen/db_images/DS_FV_tfs_v4'
PASTA     = 'DS_FV_tfs_v4'
TRANSFERS = 8
SPLITS    = ['train', 'val', 'test']
EXCLUIR   = ['.dash_stats_cache.json', '_baldes_tmp/**', '*.part']

# ==============================================================================
# 2. FUNÇÕES
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


def rclone(*args):
    """Roda rclone e devolve stdout (lança CalledProcessError se falhar)."""
    return subprocess.run(['rclone', *args], check=True,
                          capture_output=True, text=True).stdout


def excl_args():
    out = []
    for e in EXCLUIR:
        out += ['--exclude', e]
    return out


def gb(n):
    return f'{n / 1024**3:.2f} GB'


def tamanho_local(raiz):
    arqs = [f for f in raiz.rglob('*') if f.is_file()
            and f.name != '.dash_stats_cache.json' and '_baldes_tmp' not in f.parts
            and not f.name.endswith('.part')]
    return sum(f.stat().st_size for f in arqs)


def contar_local(raiz):
    return {s: len(list((raiz / s).glob('*.tfrecord.gz'))) for s in SPLITS}


def contar_drive(destino):
    out = {}
    for s in SPLITS:
        try:
            txt = rclone('lsf', f'{destino}/{s}', '--include', '*.tfrecord.gz')
            out[s] = len([l for l in txt.splitlines() if l.strip()])
        except subprocess.CalledProcessError:
            out[s] = 0
    return out


def eh_service_account(remote):
    """True se o remote usa service account (não pode subir para o Meu Drive)."""
    try:
        cfg = json.loads(rclone('config', 'dump')).get(remote, {})
    except Exception:
        return False
    return bool(cfg.get('service_account_file') or cfg.get('service_account_credentials'))

# ==============================================================================
# 3. PIPELINE
# ==============================================================================

def main():
    if shutil.which('rclone') is None:
        sys.exit('❌ rclone não instalado (sudo pacman -S rclone / sudo apt install rclone)')

    remotes = [r.rstrip(':') for r in rclone('listremotes').split()]
    if not remotes:
        sys.exit('❌ Nenhum remote rclone. Crie com: rclone config (ver checklist).')

    print('=' * 60)
    print('Parâmetros de entrada (Enter = padrão)')
    print('=' * 60)
    print('Remotes (NOME dado no rclone config, não o e-mail):')
    for k, r in enumerate(remotes, 1):
        print(f'  {k}) {r}')
    remote = ask('Remote da conta do Colab (nome ou número)',
                 REMOTE if REMOTE in remotes else remotes[0]).rstrip(':')
    if remote.isdigit() and 1 <= int(remote) <= len(remotes):
        remote = remotes[int(remote) - 1]
    local     = Path(ask('Pasta local do dataset', LOCAL)).expanduser()
    pasta     = ask('Pasta de destino no Drive (relativa ao Meu Drive)', PASTA)
    transfers = ask('Uploads em paralelo', TRANSFERS, int)
    dry_run   = ask_sim_nao('Dry-run (só mostrar o que subiria)?', False)

    # --- 1. conferências ------------------------------------------------------
    if remote not in remotes:
        sys.exit(f"❌ Remote '{remote}' não existe.")
    if eh_service_account(remote):
        sys.exit(f"❌ '{remote}' usa service account e não consegue subir para o Meu Drive.\n"
                 f'   Corrija: rclone config update {remote} service_account_file ""\n'
                 f'            rclone config reconnect {remote}:')
    if not local.is_dir():
        sys.exit(f'❌ Pasta local não encontrada: {local}')
    if (local / '_baldes_tmp').exists():
        sys.exit('❌ Existe _baldes_tmp/ na pasta: o remontar_dataset_tfrecord.py não terminou.')

    n_local = contar_local(local)
    if not sum(n_local.values()):
        sys.exit(f'❌ Nenhum *.tfrecord.gz em {local}/{{train,val,test}}')
    tam = tamanho_local(local)
    print(f'\nLocal: {local}  ({gb(tam)})')
    for s, n in n_local.items():
        print(f'  {s}: {n} arquivos')
    if (local / 'manifesto.json').exists():
        man = json.loads((local / 'manifesto.json').read_text(encoding='utf-8'))
        print('  manifesto: ' + ', '.join(
            f'{s}={sum(v for k, v in c.items() if k in ("v3", "revisao"))} patches'
            for s, c in man.get('registros', {}).items() if s in SPLITS))

    # --- 2. espaço no Drive ---------------------------------------------------
    destino = f'{remote}:{pasta}'
    try:
        about = json.loads(rclone('about', f'{remote}:', '--json'))
        livre = about.get('free')
        print(f'\nDrive ({remote}): usado {gb(about.get("used", 0))}'
              + (f' | livre {gb(livre)}' if livre is not None else ''))
        if livre is not None and livre < tam:
            print(f'⚠️  Espaço livre ({gb(livre)}) menor que o dataset ({gb(tam)}).'
                  ' Libere espaço (ex.: apagar a v3 do Drive) antes de subir.')
            if not ask_sim_nao('Continuar mesmo assim?', False):
                return
    except subprocess.CalledProcessError as e:
        sys.exit(f'❌ Não consegui acessar o Drive pelo remote {remote}:\n{e.stderr.strip()}\n'
                 f'   Refaça o login: rclone config reconnect {remote}:')

    print(f'\nResumo: {local} → {destino} | transfers={transfers} | dry_run={dry_run}')
    if not ask_sim_nao('Confirmar e continuar?', True):
        return

    # --- 3. upload ------------------------------------------------------------
    args = ['rclone', 'copy', str(local), destino, *excl_args(),
            '--transfers', str(transfers), '--checkers', '16',
            '--drive-chunk-size', '64M',
            '--retries', '5', '--low-level-retries', '20',
            '--stats', '10s', '--stats-one-line', '-P']
    if dry_run:
        args.append('--dry-run')
    rc = subprocess.run(args).returncode
    if dry_run:
        return
    if rc != 0:
        print(f'\n⚠️  rclone terminou com código {rc}. Rode de novo: ele só sobe o que faltou.')
        return

    # --- 4. verificação -------------------------------------------------------
    print('\nVerificando (rclone check --one-way)...')
    chk = subprocess.run(['rclone', 'check', str(local), destino, '--one-way', *excl_args()],
                         capture_output=True, text=True)
    resumo = [l for l in chk.stderr.splitlines() if 'differences' in l or 'ERROR' in l]
    print('  ' + ('\n  '.join(resumo[-5:]) if resumo else chk.stderr.strip()[-300:]))

    n_drive = contar_drive(destino)
    print(f'\n  {"split":<6} {"local":>7} {"drive":>7}')
    ok = chk.returncode == 0
    for s in SPLITS:
        bate = n_local[s] == n_drive[s]
        ok &= bate
        print(f'  {s:<6} {n_local[s]:>7} {n_drive[s]:>7}  {"✅" if bate else "❌"}')

    if not ok:
        print('\n⚠️  Há diferenças. Rode o script de novo para completar o upload.')
        return

    # --- 5. notebook ----------------------------------------------------------
    print('\n✅ Upload completo e conferido. No train_fotovoltaica_colab_v2.ipynb (célula 12):')
    print(f"   DATASET_ROOT_DS = '/content/drive/MyDrive/{pasta}'")
    print(f"   OUTPUT_DIR      = '/content/drive/MyDrive/TRAINING_OUTPUTS_FV_{pasta.split('_')[-1].upper()}'")
    print('   (opcional, mais rápido) copiar para o disco do Colab antes do treino:')
    print(f"   !rsync -a --info=progress2 /content/drive/MyDrive/{pasta}/ /content/{pasta}/")
    print(f"   DATASET_ROOT_DS = '/content/{pasta}'")


if __name__ == '__main__':
    main()
