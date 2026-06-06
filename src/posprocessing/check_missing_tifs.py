#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_missing_tifs.py
=====================
Lista combinações {region_id}_{ano}.tif ausentes em uma pasta local.

Uso:
  python check_missing_tifs.py --img-dir ~/teste_dash/mosaicos_tif/dataset_fotovoltaica_TIFreg
  python check_missing_tifs.py --img-dir /srv/almacen/db_images/dataset_fotovoltaica_TIFreg
"""

import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Regiões esperadas (91 ids)
# ---------------------------------------------------------------------------
EXPECTED_REGIONS = [
    '00000000000000000000', '00000000000000000001', '00000000000000000002',
    '00000000000000000003', '00000000000000000004', '00000000000000000005',
    '00000000000000000006', '00000000000000000007', '00000000000000000008',
    '00000000000000000009', '0000000000000000000a', '0000000000000000000b',
    '0000000000000000000c', '0000000000000000000d', '0000000000000000000e',
    '0000000000000000000f', '00000000000000000010', '00000000000000000011',
    '00000000000000000012', '00000000000000000013', '00000000000000000014',
    '00000000000000000015', '00000000000000000016', '00000000000000000017',
    '00000000000000000018', '00000000000000000019', '0000000000000000001a',
    '0000000000000000001b', '0000000000000000001c', '0000000000000000001d',
    '0000000000000000001e', '0000000000000000001f', '00000000000000000020',
    '00000000000000000021', '00000000000000000022', '00000000000000000023',
    '00000000000000000024', '00000000000000000025', '00000000000000000026',
    '00000000000000000027', '00000000000000000028', '00000000000000000029',
    '0000000000000000002a', '0000000000000000002b', '0000000000000000002c',
    '0000000000000000002d', '0000000000000000002e', '0000000000000000002f',
    '00000000000000000030', '00000000000000000031', '00000000000000000032',
    '00000000000000000033', '00000000000000000034', '00000000000000000035',
    '00000000000000000036', '00000000000000000037', '00000000000000000038',
    '00000000000000000039', '0000000000000000003a', '0000000000000000003b',
    '0000000000000000003c', '0000000000000000003d', '0000000000000000003e',
    '0000000000000000003f', '00000000000000000040', '00000000000000000041',
    '00000000000000000042', '00000000000000000043', '00000000000000000044',
    '00000000000000000045', '00000000000000000046', '00000000000000000047',
    '00000000000000000048', '00000000000000000049', '0000000000000000004a',
    '0000000000000000004b', '0000000000000000004c', '0000000000000000004d',
    '0000000000000000004e', '0000000000000000004f', '00000000000000000050',
    '00000000000000000051', '00000000000000000052', '00000000000000000053',
    '00000000000000000054', '00000000000000000055', '00000000000000000056',
    '00000000000000000057', '00000000000000000058', '00000000000000000059'
]

EXPECTED_YEARS = list(range(2016, 2026))  # 2016–2025

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
_ap = argparse.ArgumentParser(description='Verifica TIFs faltantes na pasta de mosaicos')
_ap.add_argument('--img-dir', type=Path, required=True,
                 help='Pasta local com os TIFs baixados do Drive')
args = _ap.parse_args()

img_dir = args.img_dir.expanduser().resolve()

if not img_dir.exists():
    print(f'Pasta não encontrada: {img_dir}')
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# Descobre arquivos presentes
# ---------------------------------------------------------------------------
present: set = set()
for f in img_dir.glob('*.tif'):
    parts = f.stem.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        present.add((parts[0], int(parts[1])))

# ---------------------------------------------------------------------------
# Calcula ausentes
# ---------------------------------------------------------------------------
expected: set = {(rid, yr) for rid in EXPECTED_REGIONS for yr in EXPECTED_YEARS}
missing        = sorted(expected - present)
unexpected     = sorted(present - expected)

total_expected = len(expected)
total_present  = len(present)
total_missing  = len(missing)

print(f'\n{"="*60}')
print(f'  Pasta    : {img_dir}')
print(f'  Esperado : {total_expected}  ({len(EXPECTED_REGIONS)} regiões × {len(EXPECTED_YEARS)} anos)')
print(f'  Presente : {total_present}')
print(f'  Faltando : {total_missing}')
print(f'{"="*60}')

if missing:
    print(f'\n[FALTANDO — {total_missing} arquivos]\n')
    # Agrupa por região para leitura mais fácil
    by_region: dict = {}
    for rid, yr in missing:
        by_region.setdefault(rid, []).append(yr)
    for rid in sorted(by_region):
        anos = ', '.join(str(y) for y in sorted(by_region[rid]))
        print(f'  {rid}  →  {anos}')
else:
    print('\nNenhum arquivo faltando. Tudo completo!')

if unexpected:
    print(f'\n[INESPERADOS — {len(unexpected)} arquivos não estavam na lista esperada]')
    for rid, yr in unexpected:
        print(f'  {rid}_{yr}')

print()
