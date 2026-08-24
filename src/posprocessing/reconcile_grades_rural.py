#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reconcile_grades_rural.py
==========================
Corrige o estado dos logs de progresso usando como ground truth a lista de
grades que foram de fato exportadas com sucesso para o Drive.

Problema:
  - Muitas grades foram "submetidas" nos logs grades_rural_<year>_*.json
    mas não salvas (erro de geometria nula ou Drive cheio).
  - load_pending_grades() pensa que estão feitas quando não estão.

Solução:
  1. Lê src/dados/list_id_grades.csv — IDs realmente exportadas (ground truth).
  2. Lê src/dados/log_rurais/grades_rural_all.json — universo completo.
  3. Move logs antigos para log_rurais/archive/ (não deleta).
  4. Cria grades_rural_<year>_reconciled.json com as IDs do CSV no formato
     esperado pelo load_submitted_ids().
  5. Imprime resumo: feitas / pendentes / fora do universo.

Uso:
  python reconcile_grades_rural.py --year 2025
  python reconcile_grades_rural.py --year 2025 --dry-run   # só mostra, não altera
"""

import sys
import json
import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

# ==============================================================================
# LOGGING
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ==============================================================================
# ARGUMENTOS
# ==============================================================================
_ap = argparse.ArgumentParser(description='Reconcilia logs de progresso com ground truth do Drive')
_ap.add_argument('--year', type=int, required=True,
                 help='Ano da exportação (ex: 2025)')
_ap.add_argument('--dry-run', action='store_true',
                 help='Apenas mostra o que seria feito, sem alterar arquivos')
_args = _ap.parse_args()

YEAR    = _args.year
DRY_RUN = _args.dry_run

# ==============================================================================
# CAMINHOS
# ==============================================================================
_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR    = _SCRIPT_DIR.parent
_LOG_DIR    = _SRC_DIR / 'dados' / 'log_rurais'
_ARCHIVE    = _LOG_DIR / 'archive'
_CSV_PATH   = _SRC_DIR / 'dados' / 'list_id_grades.csv'
_ALL_JSON   = _LOG_DIR / 'grades_rural_all.json'


def load_universe() -> dict:
    """Carrega grades_rural_all.json. Retorna dict {system_index: quantidade}."""
    if not _ALL_JSON.exists():
        log.error(f'Universo não encontrado: {_ALL_JSON}')
        log.error('Execute primeiro: python setup_grades_rural.py')
        sys.exit(1)
    entries = json.loads(_ALL_JSON.read_text(encoding='utf-8'))
    universe = {e['system_index']: e['quantidade'] for e in entries}
    log.info(f'Universo carregado: {len(universe)} grades — {_ALL_JSON.name}')
    return universe


def load_csv_ids() -> set:
    """Carrega list_id_grades.csv. Retorna set de system_index confirmados."""
    if not _CSV_PATH.exists():
        log.error(f'CSV não encontrado: {_CSV_PATH}')
        sys.exit(1)
    df = pd.read_csv(_CSV_PATH)
    # coluna pode chamar 'id_grades' ou similar; pega a primeira não-index
    col = [c for c in df.columns if 'id' in c.lower() or 'grade' in c.lower()]
    if not col:
        col = [df.columns[-1]]
    ids = set(df[col[0]].astype(str).str.strip())
    log.info(f'IDs confirmados no CSV: {len(ids)} — coluna "{col[0]}"')
    return ids


def archive_old_logs(year: int):
    """Move todos os grades_rural_<year>_*.json para archive/."""
    pattern   = f'grades_rural_{year}_*.json'
    log_files = sorted(_LOG_DIR.glob(pattern))
    if not log_files:
        log.info('Nenhum log antigo encontrado para arquivar.')
        return
    if not DRY_RUN:
        _ARCHIVE.mkdir(parents=True, exist_ok=True)
    log.info(f'Arquivando {len(log_files)} log(s) antigo(s) → {_ARCHIVE}/')
    for lf in log_files:
        dest = _ARCHIVE / lf.name
        if DRY_RUN:
            log.info(f'  [dry-run] mv {lf.name} → archive/')
        else:
            shutil.move(str(lf), str(dest))
            log.info(f'  Arquivado: {lf.name}')


def write_reconciled_log(done_entries: list, year: int):
    """Salva grades_rural_<year>_reconciled.json com os IDs confirmados."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path  = _LOG_DIR / f'grades_rural_{year}_reconciled_{timestamp}.json'
    if DRY_RUN:
        log.info(f'[dry-run] Criaria: {out_path.name} com {len(done_entries)} entradas')
        return
    out_path.write_text(
        json.dumps(done_entries, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    log.info(f'Log reconciliado salvo: {out_path.name}  ({len(done_entries)} grades)')


def main():
    log.info('=' * 60)
    log.info(f'Ano      : {YEAR}')
    log.info(f'Dry-run  : {DRY_RUN}')
    log.info(f'Log dir  : {_LOG_DIR}')
    log.info('=' * 60)

    universe   = load_universe()          # {system_index: quantidade}
    csv_ids    = load_csv_ids()           # set de IDs confirmados

    # IDs do CSV que existem no universo → feitas com sucesso
    done_in_universe   = csv_ids & set(universe.keys())
    # IDs do CSV que NÃO estão no universo (inesperado, só para diagnóstico)
    done_out_universe  = csv_ids - set(universe.keys())
    # IDs do universo que NÃO estão no CSV → ainda pendentes
    pending_ids        = set(universe.keys()) - csv_ids

    log.info('=' * 60)
    log.info(f'Universo total          : {len(universe)}')
    log.info(f'Confirmadas no CSV      : {len(csv_ids)}')
    log.info(f'  ↳ dentro do universo  : {len(done_in_universe)}')
    log.info(f'  ↳ fora do universo    : {len(done_out_universe)}  (ignoradas)')
    log.info(f'Pendentes (a exportar)  : {len(pending_ids)}')
    log.info('=' * 60)

    if done_out_universe:
        log.warning(f'IDs no CSV mas fora do universo ({len(done_out_universe)}):')
        for sid in sorted(done_out_universe)[:10]:
            log.warning(f'  {sid}')
        if len(done_out_universe) > 10:
            log.warning(f'  ... e mais {len(done_out_universe) - 10}')

    # Monta entradas do log reconciliado (mesmo formato dos logs de submissão)
    done_entries = [
        {
            'system_index': sid,
            'quantidade':   universe[sid],
            'year':         YEAR,
            'task_name':    f'{sid}_{YEAR}',
        }
        for sid in sorted(done_in_universe)
    ]

    # Arquiva logs antigos e grava log reconciliado
    archive_old_logs(YEAR)
    write_reconciled_log(done_entries, YEAR)

    if pending_ids:
        log.info(f'Próximo passo — submeter {len(pending_ids)} grade(s) pendente(s):')
        log.info(f'  python export_mosaicTIF_GEEtoDrive_rural.py --drive-folder <pasta> --year {YEAR} --limit <N>')
    else:
        log.info('Todas as grades do universo foram exportadas com sucesso!')

    log.info('Reconciliação concluída.')


if __name__ == '__main__':
    main()
