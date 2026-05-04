# Mapeamento de Sistemas Fotovoltaicos — MapBiomas

Pipeline completo para detecção e mapeamento de usinas e sistemas fotovoltaicos em zonas rurais usando imagens **Planet NICFI** (4.77 m/pixel) e segmentação semântica com **UNet + ResNet101**.

---

## Visão Geral

```
GEE (Planet NICFI)
      │
      ├─── TREINO ─────────────────────────────────────────────────────────┐
      │    download_dataset_tfrecord_fotovoltaica.py  (patches + labels)   │
      │    split_tfrecord_dataset.py                  (train/val/test)     │
      │    train_fotovoltaica_colab.ipynb             (Google Colab GPU)   │
      │                                                                    │
      └─── INFERÊNCIA ─────────────────────────────────────────────────────┤
           download_dataset_predict_fotovoltaica.py   (patches sem label)  │
                │                                                          │
                ├── [A] makePredict_patchinServer_v2.py  (modo npy)        │
                │                                                          │
                └── [B] convert_npy_to_tfrecord_fotovoltaica.py           │
                         makePredict_patchinServer_v2.py  (modo tfrecord)  │
                                                                           │
      PÓS-PROCESSAMENTO ──────────────────────────────────────────────────-┘
           join_convert_npytoTIF.py
           uploadTIF_from_localFolder_GoogleStorage.py
           transferTIF_fromGCSbucket_toGEEasset.py
```

---

## Pré-requisitos

```bash
pip install earthengine-api tensorflow numpy tqdm rasterio
```

Autenticação no Google Earth Engine:

```bash
earthengine authenticate
```

Configure a conta do projeto GEE em `src/configure_account_projects_ee.py` antes de rodar qualquer script que acesse o GEE.

---

## Estrutura de Diretórios

```
src/
├── configure_account_projects_ee.py   ← configuração de conta GEE (editar antes de usar)
├── preprocessing/
│   ├── download_dataset_tfrecord_fotovoltaica.py   ← coleta de treino (GEE → Drive)
│   └── split_tfrecord_dataset.py                   ← divisão train/val/test
├── processClass/
│   ├── segmentation_model_factory.py               ← arquitetura UNet + backbones
│   ├── train_fotovoltaica_colab.ipynb              ← treinamento (Google Colab)
│   └── makePredict_patchinServer_v2.py             ← inferência local
└── posprocessing/
    ├── download_dataset_predict_fotovoltaica.py    ← coleta de inferência (GEE → NPY)
    ├── convert_npy_to_tfrecord_fotovoltaica.py     ← conversão NPY → TFRecord
    ├── join_convert_npytoTIF.py                    ← predições NPY → GeoTIFF
    ├── uploadTIF_from_localFolder_GoogleStorage.py ← upload GCS
    └── transferTIF_fromGCSbucket_toGEEasset.py     ← GCS → GEE asset
```

---

## Passo a Passo

### FASE 1 — Coleta de Dados para Treino

**Script:** `src/preprocessing/download_dataset_tfrecord_fotovoltaica.py`

Exporta patches **257×257×8 + label** (TFRecord comprimido) direto para o Google Drive via `ee.batch.Export`. Dois pipelines internos:

- **Pipeline A** — grade regular nas regiões buffer 5 km
- **Pipeline B** — patches extras de balanceamento dentro das áreas fotovoltaicas confirmadas

```bash
# Ajuste YEARS e REGION_INIC/REGION_END no cabeçalho do script, depois:
cd src/preprocessing
python download_dataset_tfrecord_fotovoltaica.py
```

> As tasks ficam na fila do GEE. Acompanhe em `code.earthengine.google.com/tasks`.

---

### FASE 2 — Divisão Train / Val / Test

**Script:** `src/preprocessing/split_tfrecord_dataset.py`

Organiza os TFRecords baixados do Drive em subpastas `train/` (80%), `val/` (10%), `test/` (10%), garantindo que todos os arquivos de uma mesma região fiquem no mesmo split (sem vazamento de dados).

```bash
python split_tfrecord_dataset.py \
    --root /caminho/para/DATASET_FOTOVOLTAICA_NICFI_TFRECORDS
```

Estrutura de saída esperada:

```
DATASET_FOTOVOLTAICA_NICFI_TFRECORDS/
  train/  *.tfrecord.gz
  val/    *.tfrecord.gz
  test/   *.tfrecord.gz
```

---

### FASE 3 — Treinamento (Google Colab)

**Notebook:** `src/processClass/train_fotovoltaica_colab.ipynb`

Treinamento no Colab com GPU A100. Faça upload de `segmentation_model_factory.py` para o Drive antes de abrir o notebook.

Configurações principais (célula 12):

| Parâmetro | Valor atual |
|---|---|
| Arquitetura | `unet` + `resnet101` |
| Loss | `focal_tversky_boundary_loss` |
| Input shape | `(256, 256, 8)` |
| Épocas | 250 |
| Batch size | 46 |
| Learning rate | 1e-5 |
| NORM_FACTOR | 10 000 (Int16 → float32 [0, 1]) |

O modelo treinado é salvo no Drive como `.keras`. Baixe-o para o servidor antes da fase de inferência.

---

### FASE 4 — Download dos Patches para Inferência

**Script:** `src/posprocessing/download_dataset_predict_fotovoltaica.py`

Baixa patches **256×256×8** (sem label) diretamente para disco local via `ee.data.computePixels()`, sem consumir tasks do GEE.

```bash
cd src/posprocessing
python download_dataset_predict_fotovoltaica.py \
    --output-dir /dados/dataset_fotovoltaica_npy

python download_dataset_predict_tfrecord_fotovoltaica.py --output-dir ~/db_images/dataset_fotovoltaica_npy --year_inic 2016 --year_end 2020
```

Parâmetros a ajustar no cabeçalho do script:

| Parâmetro | Descrição |
|---|---|
| `ASSET_REGIONS` | FeatureCollection com as regiões a mapear |
| `YEARS` | Lista de anos (ex.: `range(2016, 2026)`) |
| `REGION_INIC` / `REGION_END` | Intervalo de regiões (para paralelizar em múltiplas sessões) |
| `STRIDE_PIXELS` | Passo da grade em pixels (padrão: 200) |

Saída:

```
/dados/dataset_fotovoltaica_npy/
  <region_id>/<year>/patch_r0000_c0000.npy   →  (256, 256, 8) int16
  <region_id>/<year>/patch_r0000_c0000.json  →  transform GDAL + metadados
```

O script retoma automaticamente patches já baixados (resume).

---

### FASE 5 — Inferência

Duas opções de pipeline, escolha conforme o volume de dados:

#### Opção A — Direto sobre NPY (recomendada para volumes moderados)

```bash
python src/processClass/makePredict_patchinServer_v2.py \
    --model-path   /modelos/best_unet_resnet101_20260502.keras \
    --input-dir    /dados/dataset_fotovoltaica_npy \
    --input-format npy \
    --output-dir   /dados/predict_fotovoltaica \
    --threshold    0.5 \
    --batch-size   8
```

#### Opção B — Via TFRecord (recomendada para grandes volumes com pipeline tf.data)

**Passo B1 — Converter NPY → TFRecord:**

```bash
python src/posprocessing/convert_npy_to_tfrecord_fotovoltaica.py
```

Ajuste `NPY_DIR` e `TFRECORD_DIR` no cabeçalho do script. Saída: shards de 64 patches em `<TFRECORD_DIR>/<region_id>/<year>/*.tfrecord`.

**Passo B2 — Inferência sobre TFRecord:**

```bash
python src/processClass/makePredict_patchinServer_v2.py \
    --model-path   /modelos/best_unet_resnet101_20260502.keras \
    --input-dir    /dados/dataset_fotovoltaica_tfrecord \
    --input-format tfrecord \
    --output-dir   /dados/predict_fotovoltaica \
    --batch-size   16
```

**Parâmetros disponíveis (`makePredict_patchinServer_v2.py`):**

| Argumento | Obrigatório | Descrição |
|---|---|---|
| `--model-path` | sim | Caminho para o `.keras` treinado |
| `--input-dir` | sim | Pasta raiz dos patches (NPY ou TFRecord) |
| `--input-format` | não | `npy` ou `tfrecord` (padrão: `npy`) |
| `--output-dir` | sim | Pasta de saída das predições |
| `--threshold` | não | Limiar salvo nos metadados (padrão: `0.5`) |
| `--batch-size` | não | Patches por batch (padrão: `8`) |
| `--years` | não | Anos a processar (ex.: `--years 2023 2024`) |
| `--regions` | não | IDs de região (ex.: `--regions 0001 0002`) |

Saída por patch:

```
/dados/predict_fotovoltaica/
  <region_id>/<year>/patch_r0000_c0000_pred.npy   →  (256, 256) float32 [0, 1]
  <region_id>/<year>/patch_r0000_c0000_pred.json  →  transform GDAL + params
```

O script retoma predições já existentes (resume).

---

### FASE 6 — Pós-processamento: NPY → GeoTIFF

**Script:** `src/posprocessing/join_convert_npytoTIF.py`

Converte os mapas de probabilidade (`_pred.npy`) para GeoTIFF georreferenciado usando o transform GDAL salvo no JSON.

```bash
python src/posprocessing/join_convert_npytoTIF.py
```

---

### FASE 7 — Upload para Google Cloud Storage e GEE

```bash
# 1. Sobe os TIFs para um bucket GCS
python src/posprocessing/uploadTIF_from_localFolder_GoogleStorage.py

# 2. Importa do GCS para asset no GEE
python src/posprocessing/transferTIF_fromGCSbucket_toGEEasset.py
```

---

## Bandas e Índices Espectrais

| Canal | Nome | Fórmula | Papel na separação |
|---|---|---|---|
| 0 | blue | B (NICFI) | base espectral |
| 1 | green | G (NICFI) | base espectral |
| 2 | red | R (NICFI) | base espectral |
| 3 | nir | N (NICFI) | base espectral |
| 4 | pvi | `(Blue − NIR) / (Blue + NIR + 1)` | alto quando Blue > NIR → separa placa de água/sombra |
| 5 | iia | `(Green − 4·NIR) / (Green + 4·NIR + 1)` | alto para alvos escuros no NIR → separa {placa, água} de {solo, vegetação} |
| 6 | ri | `2.4·(Red − Green) / (Red + Green + 1)` | alto para alvos avermelhados → separa solo exposto dentro dos escuros |
| 7 | evi | `2.4·(NIR − Red) / (1 + NIR + Red)` | sensibilidade a vegetação densa |

Todos os valores são normalizados para `Int16 [0, 10 000]` no GEE e divididos por `10 000` antes da inferência.

**Lógica de separação em cascata:**

```
NIR alto? → solo / vegetação  (IIA baixo)
  ├── RI alto?  → solo exposto / área queimada
  └── RI baixo? → vegetação

NIR baixo? → alvo escuro       (IIA alto)
  ├── PVI alto? → placa solar  (Blue relativamente alto)
  └── PVI baixo? → água / sombra
```

---

## Arquitetura do Modelo

- **Cabeça:** UNet com atenção (canal + espacial + focal)
- **Backbone:** ResNet101 (sem pesos pré-treinados, treinado do zero em 8 canais)
- **Loss:** `focal_tversky_boundary_loss` — Focal Tversky (α=0.3, β=0.7, γ=1.25) + Boundary Loss (peso 0.85)
- **Métricas:** Dice, IoU, Precision, Recall
- **Parâmetros:** ~34.7 M

---

## Notas Importantes

- **Limite de tasks GEE (3 000):** a fase de coleta de treino (Fase 1) consome tasks; a fase de inferência (Fase 4) usa `computePixels` e **não** consome tasks.
- **Resume automático:** os scripts de download (Fase 4) e inferência (Fase 5) pulam arquivos já existentes em disco.
- **Paralelização:** divida `REGION_INIC` / `REGION_END` entre múltiplas sessões para acelerar o download.
- **GPU:** a inferência funciona em CPU mas é significativamente mais lenta; recomenda-se GPU com ≥ 8 GB de VRAM para `--batch-size 8`.
