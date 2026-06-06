#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scripts para listar os Ids e imprimir na tela
"""

import sys
import os

import ee
from pathlib import Path

pathparent = str(Path(os.getcwd()).parents[0])
sys.path.append(pathparent)
from configure_account_projects_ee import get_current_account

projAccount = get_current_account()
print(f"Projeto selecionado >>> {projAccount} <<<")

try:
    ee.Initialize(project=projAccount)
    print('Earth Engine inicializado com sucesso!')
except Exception as e:
    print("Erro de Inicialização:", e)
    raise

# ==============================================================================
# 1. CONFIGURAÇÕES
# ==============================================================================

# ── Dados da regions ─────────────────────────────────────────────────────────────

ASSET_REGIONS_2024 = "projects/mapbiomas-arida/energias/shp_area_fotovoltaic_samples_update_16_05_2026" 
regions = ee.FeatureCollection(ASSET_REGIONS_2024)
lst_id_regions = regions.reduceColumns(ee.Reducer.toList(), ['system:index']).get('list').getInfo()
print(f"we have {len(lst_id_regions)} regions")
print(lst_id_regions)
