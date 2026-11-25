#!/usr/bin/env python3
"""
Baixa TFRecords do Google Drive para o servidor I7 (via rclone)
===============================================================
Copia a pasta DS_FV_tfs_scaled_v3 (usada em train_fotovoltaica_colab_v2.ipynb)
do Drive para o servidor, onde já estão as amostras novas (tfrecord_fv_revisao),
para depois embaralhar tudo junto.

- Mantém a estrutura train/ val/ test/.
- Usa `rclone copy`: arquivos que já existem iguais no destino são pulados
  (rodar de novo retoma); nada é apagado no Drive nem no servidor.

Pré-requisito (uma vez): um remote rclone do tipo "drive" logado na conta
Google DONA da pasta (a mesma do Colab). No servidor sem navegador:

    rclone config
      n (new) → nome: drive_fv → storage: drive → scope: 2 (read-only)
      "Use web browser to automatically authenticate?" → n
      rode o "rclone authorize drive ..." que ele mostrar numa máquina com
      navegador, faça login e cole o token de volta no servidor.

Se a pasta foi COMPARTILHADA com a conta do remote (não está no "Meu Drive"
dela), responda "s" em "Pasta está em 'Compartilhados comigo'?".

Uso:
    python src/preprocessing/baixar_tfrecords_drive_rclone.py
    Parâmetros pedidos via input(); Enter aceita o [padrão].
"""

import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ==============================================================================
# 1. CONFIGURAÇÕES (padrões das perguntas)
# ==============================================================================
REMOTE    = 'drive_fv'
PASTA     = 'DS_FV_tfs_scaled_v3'
DESTINO   = '~/almacen/db_images/DS_FV_tfs_scaled_v3'
TRANSFERS = 8
SPLITS    = ['train', 'val', 'test']
PADRAO    = '*.tfrecord.gz'

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


def rclone(*args, capture=True):
    """Roda rclone; com capture=True devolve stdout (lança erro se falhar)."""
    cmd = ['rclone', *args]
    if capture:
        return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    return subprocess.run(cmd).returncode


def contar_origem(origem, opt_shared):
    for split in SPLITS:
        try:
            out = rclone('lsf', *opt_shared, f'{origem}/{split}', '--include', PADRAO)
            n = len([l for l in out.splitlines() if l.strip()])
        except subprocess.CalledProcessError:
            n = 0
        print(f'  {split}: {n}')
    try:
        out = rclone('size', *opt_shared, origem, '--include', PADRAO)
        for linha in out.strip().splitlines():
            print(f'  {linha}')
    except subprocess.CalledProcessError:
        pass


def contar_destino(destino):
    for split in SPLITS:
        n = len(list((destino / split).glob(PADRAO)))
        print(f'  {split}: {n}')

# ==============================================================================
# 3. PIPELINE
# ==============================================================================

def main():
    if shutil.which('rclone') is None:
        sys.exit('❌ rclone não instalado (sudo pacman -S rclone  /  sudo apt install rclone)')

    remotes = [r.rstrip(':') for r in rclone('listremotes').split()]
    print('=' * 60)
    print('Remotes rclone disponíveis:', ', '.join(remotes) or '(nenhum)')
    print('Parâmetros de entrada (Enter = padrão)')
    print('=' * 60)

    remote    = ask('Remote rclone da conta dona da pasta', REMOTE).rstrip(':')
    pasta     = ask('Pasta no Drive (relativa ao Meu Drive)', PASTA)
    shared    = ask_sim_nao("Pasta está em 'Compartilhados comigo'?", False)
    destino   = Path(ask('Pasta de destino no servidor', DESTINO)).expanduser()
    transfers = ask('Transferências em paralelo', TRANSFERS, int)
    dry_run   = ask_sim_nao('Dry-run (só listar o que seria copiado)?', False)

    if remote not in remotes:
        sys.exit(f"❌ Remote '{remote}:' não existe. Crie com: rclone config "
                 '(ver docstring do script).')

    opt_shared = ['--drive-shared-with-me'] if shared else []
    origem     = f'{remote}:{pasta}'

    print(f'\nConferindo origem {origem} ...')
    try:
        rclone('lsf', *opt_shared, origem, '--max-depth', '1')
    except subprocess.CalledProcessError as e:
        sys.exit(f"❌ Não achei '{pasta}' em '{remote}:'. Confira a conta do remote "
                 f"/ opção de compartilhado.\n{e.stderr.strip()}")

    print(f'Arquivos {PADRAO} na origem por split:')
    contar_origem(origem, opt_shared)

    print(f'\nResumo: {origem} → {destino} | transfers={transfers} '
          f'| compartilhado={shared} | dry_run={dry_run}')
    if not ask_sim_nao('Confirmar e continuar?', True):
        return

    destino.mkdir(parents=True, exist_ok=True)
    log = destino / f'rclone_copy_{datetime.now():%Y%m%d_%H%M%S}.log'

    args = ['copy', origem, str(destino),
            '--include', PADRAO,
            '--transfers', str(transfers),
            '--checkers', '16',
            '--drive-acknowledge-abuse',
            '--retries', '5', '--low-level-retries', '20',
            '--stats', '10s', '--stats-one-line', '-P',
            '--log-file', str(log), '--log-level', 'INFO',
            *opt_shared]
    if dry_run:
        args.append('--dry-run')

    rc = rclone(*args, capture=False)       # progresso aparece direto no terminal
    if rc != 0:
        print(f'\n⚠️  rclone terminou com código {rc}. Veja o log: {log}')
        print('   Rodar de novo baixa só o que faltou.')

    print('\nArquivos no destino por split:')
    contar_destino(destino)
    print(f'Log: {log}')
    print('Concluído.')


if __name__ == '__main__':
    main()
