# -*- coding: utf-8 -*-
"""
Verifica quais grades com sistemas fotovoltaicos rurais ainda NÃO têm TIF salvo
e grava a lista em CSV, para depois re-exportar / baixar só as que faltam.

Feito para rodar no Google Colab (células separadas por "# %%").

Lógica
------
1. Grades esperadas = grades de ASSET_GRADE que intersectam ASSET_PHOTOVOLTAIC
   (mesma regra de src/posprocessing/setup_grades_rural.py), com 'quantidade'
   de pontos FV por grade.
2. TIFs existentes = arquivos na pasta (Drive montado) e/ou numa listagem .txt
   gerada no servidor (ex.: `ls ~/almacen/db_images/tif_planet_2025 > lista_tifs.txt`).
   Nome esperado do TIF (export_mosaicTIF_GEEtoDrive_rural.py):
       {system_index}_{year}.tif
   ou, quando o GEE divide a imagem:
       {system_index}_{year}-0000000000-0000000000.tif
3. Faltantes = esperadas - existentes  →  CSV.
"""

# %% [1] Instalação / autenticação
# !pip install -q earthengine-api   # já vem no Colab
import os
import re
import ee
import pandas as pd

PROJECT_EE = 'mapbiomas-arida'          # projeto GEE usado para autenticar
ee.Authenticate()
ee.Initialize(project=PROJECT_EE)

from google.colab import drive
drive.mount('/content/drive')

# %% [2] Parâmetros
ASSET_GRADE        = 'projects/nexgenmap/SAD_MapBiomas/DL/SHP_grades_BR_35pathces_AllBrV3'
ASSET_PHOTOVOLTAIC = 'projects/mapbiomas-arida/fotovoltaic_rural'

YEAR = 2025          # ano usado no nome do TIF ({system_index}_{YEAR}.tif); None = ignora ano

# Pasta(s) onde estão os TIFs. Deixe vazia se for usar só a listagem .txt.
TIF_DIRS = [
    '/content/drive/MyDrive/DS_FV_TIFs_scaled/PREDICT_V2',
]

# Listagem opcional vinda do servidor local (um nome de arquivo por linha).
# No servidor:  ls ~/almacen/db_images/tif_planet_2025 > lista_tifs.txt
# e suba o arquivo para o Drive.
LISTA_TXT = '/content/drive/MyDrive/lista_tifs.txt'

OUT_CSV = '/content/drive/MyDrive/grades_faltantes_fv_rural.csv'

# %% [3] Grades esperadas (GEE)
colection_fv = ee.FeatureCollection(ASSET_PHOTOVOLTAIC)
grid_shp     = ee.FeatureCollection(ASSET_GRADE)
grades       = grid_shp.filterBounds(colection_fv)

def add_quantity(feat):
    return feat.set('quantidade', colection_fv.filterBounds(feat.geometry()).size())

features = grades.map(add_quantity).select(['quantidade']).getInfo()['features']

df_grades = pd.DataFrame([
    {'system_index': f['id'], 'quantidade': f['properties'].get('quantidade', 0)}
    for f in features
])
print(f'Grades com FV (esperadas): {len(df_grades)}')

# %% [4] TIFs existentes
TIF_RE = re.compile(r'\.tiff?$', re.IGNORECASE)
# remove sufixo de tile do GEE: -0000000000-0000000000
TILE_RE = re.compile(r'-\d{10}-\d{10}$')
YEAR_RE = re.compile(rf'_{YEAR}$') if YEAR else re.compile(r'_\d{4}$')

def nome_para_grade(nome: str) -> str | None:
    nome = os.path.basename(nome.strip())
    if not TIF_RE.search(nome):
        return None
    base = TIF_RE.sub('', nome)
    base = TILE_RE.sub('', base)
    if not YEAR_RE.search(base):
        return None                 # ano diferente do pedido
    return YEAR_RE.sub('', base)

nomes = []
for d in TIF_DIRS:
    if os.path.isdir(d):
        for root, _, files in os.walk(d):
            nomes.extend(files)
        print(f'{d}: {sum(len(f) for _, _, f in os.walk(d))} arquivos')
    else:
        print(f'⚠️  Pasta não encontrada: {d}')

if LISTA_TXT and os.path.isfile(LISTA_TXT):
    with open(LISTA_TXT, encoding='utf-8') as fh:
        linhas = [l for l in fh if l.strip()]
    nomes.extend(linhas)
    print(f'{LISTA_TXT}: {len(linhas)} linhas')

existentes = {g for g in map(nome_para_grade, nomes) if g}
print(f'TIFs reconhecidos: {len(existentes)} grades distintas')
print('Exemplos:', sorted(existentes)[:5])

# %% [5] Comparação e CSV
esperadas = set(df_grades['system_index'])
faltam    = esperadas - existentes
sobram    = existentes - esperadas      # TIF sem grade correspondente (nome estranho / ano / asset diferente)

df_falt = (df_grades[df_grades['system_index'].isin(faltam)]
           .sort_values('quantidade', ascending=False)
           .assign(year=YEAR,
                   task_name=lambda d: d['system_index'] + (f'_{YEAR}' if YEAR else '')))

df_falt.to_csv(OUT_CSV, index=False)

print(f'Esperadas : {len(esperadas)}')
print(f'Salvas    : {len(esperadas & existentes)}')
print(f'Faltam    : {len(faltam)}  →  {OUT_CSV}')
print(f'Sem grade : {len(sobram)}  (ex.: {sorted(sobram)[:5]})')
df_falt.head(20)
