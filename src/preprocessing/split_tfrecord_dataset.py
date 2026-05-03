#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Split TFRecord Dataset — Fotovoltaica
======================================
Organiza os arquivos TFRecord exportados pelo Earth Engine em subpastas
  train/ (80%)  |  val/ (10%)  |  test/ (10%)

Estratégia de split por REGIÃO (não por arquivo):
  Todos os arquivos de uma mesma região sempre ficam no mesmo split,
  evitando vazamento de dados entre treino e validação/teste.

Uso:
    python split_tfrecord_dataset.py

    # Ou passando o caminho diretamente:
    python split_tfrecord_dataset.py --root /caminho/para/DATASET_FOTOVOLTAICA_NICFI_TFRECORDS

Padrão de nome esperado:
    tfrecord_fv_{region_id}_{year}_part{nnn}[.tfrecord]

Exemplo real:
    tfrecord_fv_00000000000000000000_2019_part000
    tfrecord_fv_00000000000000000009_2022_part000
"""

import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


# ==============================================================================
# Configurações
# ==============================================================================

DEFAULT_ROOT   = '/run/media/superuser/Almacen/imgDB/DATASET_FOTOVOLTAICA_NICFI_TFRECORDS'
TRAIN_RATIO    = 0.80
VAL_RATIO      = 0.10
TEST_RATIO     = 0.10
RANDOM_SEED    = 42
COPY_MODE      = False   # True → copia (mantém originais) | False → move
VALID_EXTS     = {'.tfrecord', '.gz', ''}   # .gz = tfrecord comprimido do GEE

# Regex para extrair region_id do nome do arquivo
# Espera: tfrecord_fv_<region_id>_<year>_part<nnn>
_FNAME_RE = re.compile(
    r'^tfrecord_fv_(?P<region>.+?)_(?P<year>\d{4})_part\d+',
    re.IGNORECASE,
)


# ==============================================================================
# Funções
# ==============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Split TFRecord dataset em train/val/test')
    p.add_argument(
        'root', type=Path, nargs='?', default=Path(DEFAULT_ROOT),
        help='Pasta raiz com os arquivos TFRecord (default: %(default)s)',
    )
    p.add_argument(
        '--train', type=float, default=TRAIN_RATIO,
        help='Proporção de treino (default: %(default).2f)',
    )
    p.add_argument(
        '--val', type=float, default=VAL_RATIO,
        help='Proporção de validação (default: %(default).2f)',
    )
    p.add_argument(
        '--test', type=float, default=TEST_RATIO,
        help='Proporção de teste (default: %(default).2f)',
    )
    p.add_argument(
        '--seed', type=int, default=RANDOM_SEED,
        help='Semente aleatória (default: %(default)d)',
    )
    p.add_argument(
        '--copy', action='store_true', default=COPY_MODE,
        help='Copia os arquivos em vez de mover (mantém originais)',
    )
    p.add_argument(
        '--dry-run', action='store_true',
        help='Apenas simula — não move/copia nenhum arquivo',
    )
    return p.parse_args()


def collect_files(root: Path) -> dict[str, list[Path]]:
    """
    Varre a pasta raiz (ignora subpastas train/val/test já existentes)
    e agrupa arquivos por region_id.

    Returns
    -------
    dict: { region_id : [Path, Path, ...] }
    """
    skip_dirs = {'train', 'val', 'test'}
    region_files: dict[str, list[Path]] = defaultdict(list)
    unrecognized: list[Path] = []

    for f in sorted(root.iterdir()):
        if f.is_dir():
            continue
        if f.suffix.lower() not in VALID_EXTS:
            continue
        if f.parent.name.lower() in skip_dirs:
            continue

        m = _FNAME_RE.match(f.stem)
        if m:
            region_files[m.group('region')].append(f)
        else:
            unrecognized.append(f)

    if unrecognized:
        print(f"\n[AVISO] {len(unrecognized)} arquivo(s) não reconhecido(s) "
              f"(padrão diferente, serão ignorados):")
        for u in unrecognized[:5]:
            print(f"  {u.name}")
        if len(unrecognized) > 5:
            print(f"  ... e mais {len(unrecognized) - 5}")

    return dict(region_files)


def split_regions(region_ids: list[str],
                  train_r: float,
                  val_r: float,
                  seed: int) -> tuple[list[str], list[str], list[str]]:
    """
    Divide a lista de region_ids em train / val / test
    garantindo que cada região aparece em apenas um split.
    """
    ids = list(region_ids)
    random.seed(seed)
    random.shuffle(ids)

    n       = len(ids)
    n_train = round(n * train_r)
    n_val   = round(n * val_r)

    train = ids[:n_train]
    val   = ids[n_train : n_train + n_val]
    test  = ids[n_train + n_val :]

    return train, val, test


def move_or_copy(src: Path, dst_dir: Path,
                 copy: bool, dry_run: bool) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name

    if dry_run:
        action = 'COPY' if copy else 'MOVE'
        print(f"  [{action}] {src.name}  →  {dst_dir.name}/")
        return

    if copy:
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), dst)


def distribute_files(region_files: dict[str, list[Path]],
                     train_ids: list[str],
                     val_ids: list[str],
                     test_ids: list[str],
                     root: Path,
                     copy: bool,
                     dry_run: bool) -> dict[str, int]:
    """Move/copia os arquivos para as subpastas corretas."""
    split_map = (
        [(rid, 'train') for rid in train_ids] +
        [(rid, 'val')   for rid in val_ids]   +
        [(rid, 'test')  for rid in test_ids]
    )

    counters = {'train': 0, 'val': 0, 'test': 0}

    for region_id, split_name in split_map:
        dst_dir = root / split_name
        for f in region_files[region_id]:
            move_or_copy(f, dst_dir, copy, dry_run)
            counters[split_name] += 1

    return counters


def print_summary(region_files: dict[str, list[Path]],
                  train_ids: list[str],
                  val_ids: list[str],
                  test_ids: list[str],
                  counters: dict[str, int],
                  dry_run: bool) -> None:
    total_regions = len(region_files)
    total_files   = sum(len(v) for v in region_files.values())

    print('\n' + '=' * 60)
    print('SPLIT SUMMARY')
    print('=' * 60)
    print(f"{'Split':<8} {'Regiões':>10} {'Arquivos':>10}  {'%':>6}")
    print('-' * 40)
    for split_name, ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
        pct = 100 * len(ids) / total_regions if total_regions else 0
        print(f"{'  ' + split_name:<8} {len(ids):>10} {counters[split_name]:>10}  {pct:>5.1f}%")
    print('-' * 40)
    print(f"{'  TOTAL':<8} {total_regions:>10} {total_files:>10}  100.0%")
    if dry_run:
        print('\n[DRY-RUN] Nenhum arquivo foi movido/copiado.')


# ==============================================================================
# Main
# ==============================================================================

def main() -> None:
    args = parse_args()

    root = args.root
    if not root.exists():
        raise FileNotFoundError(
            f"Pasta não encontrada: {root}\n"
            f"Ajuste DEFAULT_ROOT no script ou use --root <caminho>"
        )

    # Valida proporções
    total = args.train + args.val + args.test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"train + val + test deve ser 1.0 (atual: {total:.4f})"
        )

    print(f"Pasta raiz   : {root}")
    print(f"Split        : train={args.train:.0%}  val={args.val:.0%}  test={args.test:.0%}")
    print(f"Modo         : {'COPY' if args.copy else 'MOVE'}"
          + (" [DRY-RUN]" if args.dry_run else ""))
    print(f"Semente      : {args.seed}")

    # 1. Coleta e agrupa por região
    print("\nVarrendo arquivos...")
    region_files = collect_files(root)
    if not region_files:
        print("Nenhum arquivo TFRecord encontrado. Verifique a pasta e o padrão de nome.")
        return

    total_files   = sum(len(v) for v in region_files.values())
    total_regions = len(region_files)
    print(f"Regiões únicas encontradas : {total_regions}")
    print(f"Arquivos TFRecord totais   : {total_files}")

    # 2. Split por região
    train_ids, val_ids, test_ids = split_regions(
        list(region_files.keys()), args.train, args.val, args.seed
    )

    print(f"\nRegiões por split  →  train: {len(train_ids)}  "
          f"| val: {len(val_ids)}  | test: {len(test_ids)}")

    # 3. Move/copia arquivos
    print()
    counters = distribute_files(
        region_files, train_ids, val_ids, test_ids,
        root, args.copy, args.dry_run,
    )

    # 4. Resumo
    print_summary(region_files, train_ids, val_ids, test_ids, counters, args.dry_run)


if __name__ == '__main__':
    main()
