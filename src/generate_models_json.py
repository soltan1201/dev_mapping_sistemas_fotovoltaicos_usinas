#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_models_json.py
=======================
Escaneia a pasta de modelos e gera um JSON com a estrutura:

  {
    "UNet": {
      "resnet50": {
        "L9": {
          "best_unet_resnet50_20260429_1436.keras": "history_unet_resnet50_20260429_1436.csv"
        },
        "L5": {
          "best_5L_unet_resnet50_20260520_0305.keras": "history_unet_resnet50_20260520_0305.csv"
        }
      }
    }
  }

Convenção de nomes esperada
---------------------------
  Keras : best_{SAT_}?{arch}_{backbone}_{YYYYMMDD}_{HHMM}.keras
  CSV   : history_{arch}_{backbone}_{YYYYMMDD}_{HHMM}.csv

  Prefixo de sobre layers utlizados nos dados de entrada (opcional):
    L5 |  9L 

Uso:
  python generate_models_json.py --models-dir /srv/almacen/models
  python generate_models_json.py --models-dir ./models --output models_index.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

# ── Mapeamentos ───────────────────────────────────────────────────────────────

LAYERS_L5_PREFIX = '5l'   # prefixo no nome do arquivo → satélite L5
DEFAULT_SATELLITE = 'L9'

ARCH_DISPLAY = {
    'unet':       'UNet',
    'deeplabv3':  'DeepLabV3',
    'segnet':     'SegNet',
    'fcn':        'FCN',
}

# Regex:  best_  [SAT_]?  ARCH  _  BACKBONE  _  DATE  _  TIME  .keras
# Grupos:         (1)     (2)       (3)         (4  +  5)
KERAS_RE = re.compile(
    r'^best_(?:([0-9]+l)_)?([a-z][a-z0-9]+)_([a-z][a-z0-9]+)_(\d{8}_\d{4})\.keras$',
    re.IGNORECASE,
)

# Regex:  history_  ARCH  _  BACKBONE  _  DATE  _  TIME  .csv
CSV_RE = re.compile(
    r'^history_([a-z][a-z0-9]+)_([a-z][a-z0-9]+)_(\d{8}_\d{4})\.csv$',
    re.IGNORECASE,
)


# ── Lógica ────────────────────────────────────────────────────────────────────

def _arch_label(raw: str) -> str:
    return ARCH_DISPLAY.get(raw.lower(), raw)


def _numero_layers_input(prefix: str | None) -> str:
    if prefix and prefix.lower() == LAYERS_L5_PREFIX:
        return 'L5'
    return DEFAULT_SATELLITE


def build_index(models_dir: Path) -> dict:
    print("We read all files in >>> ", models_dir)
    keras_files = sorted(models_dir.glob('*.keras'))
    csv_files   = sorted(models_dir.glob('*.csv'))
    print(f"The file have {len(csv_files)} file .CSVs")

    # Índice CSV por (arch, backbone, timestamp)
    csv_index: dict[tuple, str] = {}
    for csv in csv_files:
        m = CSV_RE.match(csv.name)
        if m:
            key = (m.group(1).lower(), m.group(2).lower(), m.group(3))
            csv_index[key] = csv.name

    index: dict = {}
    unmatched: list[str] = []

    for keras in keras_files:
        m = KERAS_RE.match(keras.name)
        if not m:
            unmatched.append(keras.name)
            continue

        sat_prefix, arch_raw, backbone_raw, timestamp = m.groups()
        arch          = _arch_label(arch_raw)
        backbone      = backbone_raw.lower()
        numero_layers = _numero_layers_input(sat_prefix)

        csv_key  = (arch_raw.lower(), backbone, timestamp)
        csv_name = csv_index.get(csv_key, None)

        # Navega / cria os níveis do dicionário
        index \
            .setdefault(arch, {}) \
            .setdefault(backbone, {}) \
            .setdefault(numero_layers, {})[keras.name] = csv_name

    if unmatched:
        print(f'[AVISO] {len(unmatched)} arquivo(s) não reconhecido(s):', file=sys.stderr)
        for name in unmatched:
            print(f'  {name}', file=sys.stderr)

    return index


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Gera JSON de índice dos modelos (.keras + .csv)')
    parser.add_argument(
        '--models-dir', type=Path, required=True,
        help='Pasta onde estão os arquivos .keras e .csv',
    )
    parser.add_argument(
        '--output', type=Path, default=None,
        help='Arquivo JSON de saída (padrão: <models-dir>/models_index.json)',
    )
    args = parser.parse_args()

    models_dir = args.models_dir.expanduser().resolve()
    if not models_dir.is_dir():
        print(f'Erro: pasta não encontrada: {models_dir}', file=sys.stderr)
        sys.exit(1)

    output = args.output or models_dir / 'models_index.json'

    index = build_index(models_dir)

    with open(output, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f'JSON salvo em: {output}')
    print(json.dumps(index, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
