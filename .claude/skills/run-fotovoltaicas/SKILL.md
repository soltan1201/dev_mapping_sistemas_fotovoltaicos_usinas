---
name: run-fotovoltaicas
description: run, start, launch, screenshot, verify, test the fotovoltaicas dashboard or streamlit viewer — Dash predict dashboard and Streamlit NPY patch viewer for the MapBiomas photovoltaic mapping pipeline
---

# run-fotovoltaicas

Dois apps interativos neste projeto:

| App | Script | Porta padrão | Propósito |
|-----|--------|-------------|-----------|
| **Dashboard Dash** | `src/dash/dashboard_predict.py` | 8051 | Visualiza mosaicos NICFI lado a lado com overlay de predição |
| **Viewer Streamlit** | `src/show_colect_data/viewer_patches_npy.py` | 8052 | Navega patches `.npy` por região/ano com visualização multicanal |

O harness de smoke está em `.claude/skills/run-fotovoltaicas/smoke.sh` — lança os dois apps, tira screenshots via Playwright e encerra.

---

## Pré-requisitos

```bash
pip install dash dash-bootstrap-components rasterio scipy pillow playwright \
    --break-system-packages
python3 -m playwright install chromium
# streamlit já instalado
```

Verificar:

```bash
python3 -c "import dash, dash_bootstrap_components, rasterio, scipy, PIL; print('ok')"
```

---

## Dados de teste

O dashboard precisa de um `predict_index.json` e de TIFs de mosaico. Para testar sem dados reais:

```bash
# 1. Cria TIFs sintéticos (256×256, 5 bandas Int16)
python3 - << 'EOF'
import numpy as np, rasterio
from rasterio.transform import from_bounds
from pathlib import Path

mosaic_dir = Path('src/bd_images/mosaicos_tif_teste')
mosaic_dir.mkdir(exist_ok=True)
pairs = [('00000000000000000000',2022),('00000000000000000000',2025),
         ('00000000000000000001',2025),('00000000000000000002',2025),
         ('00000000000000000006',2025),('0000000000000000000a',2025)]
t = from_bounds(0,0,1,1,256,256)
for rid, yr in pairs:
    rng = np.random.default_rng(int(rid,16)+yr)
    data = rng.integers(200,8000,(5,256,256),dtype=np.int16)
    with rasterio.open(mosaic_dir/f'{rid}_{yr}.tif','w',driver='GTiff',
        height=256,width=256,count=5,dtype='int16',crs='EPSG:4326',transform=t) as dst:
        dst.write(data)
print('TIFs sintéticos criados em', mosaic_dir)
EOF

# 2. Cria predict_index.json apontando para os TIFs sintéticos e as predições existentes
cat > src/bd_images/predict_index_test.json << 'JSON'
{
  "mosaicos": "/home/superuser/Dados/mapbiomas/dev_mapping_sistemas_fotovoltaicos_usinas/src/bd_images/mosaicos_tif_teste",
  "saidas": {
    "unet_efficientnetb7": "/home/superuser/Dados/mapbiomas/dev_mapping_sistemas_fotovoltaicos_usinas/src/bd_images/mosaicos_tif_unet_efficientnetb7"
  }
}
JSON
# substituir /home/superuser/Dados/mapbiomas/dev_mapping_sistemas_fotovoltaicos_usinas pelo caminho absoluto do projeto
```

Com dados reais, o `predict_index.json` é gerado automaticamente pelo `makePredict_fromTIF.py`:

```bash
python3 src/processClass/makePredict_fromTIF.py \
    --model-path models/best_5L_unet_efficientnetb7_20260520_2241.keras \
    --input-dir  ~/dados/mosaicos_tif
# → cria ~/dados/predict_index.json automaticamente
```

---

## Run (agent path) — smoke script

```bash
bash .claude/skills/run-fotovoltaicas/smoke.sh
# saída: /tmp/dash_smoke.png  /tmp/streamlit_smoke.png
```

Opções:

```bash
bash .claude/skills/run-fotovoltaicas/smoke.sh 8051 8052   # portas customizadas
```

O script:
1. Sobe o dashboard Dash (`src/dash/dashboard_predict.py`) em background
2. Sobe o viewer Streamlit (`src/show_colect_data/viewer_patches_npy.py`) em background
3. Aguarda 8 s de inicialização
4. Tira screenshots headless via Playwright
5. Encerra ambos os processos

---

## Run (agent path) — manual

### Dashboard Dash

```bash
python3 src/dash/dashboard_predict.py \
    --index-file src/bd_images/predict_index_test.json \
    --port 8051
```

Curl de verificação:

```bash
curl -s -o /dev/null -w "%{http_code}" http://localhost:8051/
# esperado: 200
```

Screenshot manual via Playwright:

```python
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    pg = b.new_page(viewport={"width":1400,"height":900})
    pg.goto("http://localhost:8051/", wait_until="networkidle", timeout=15000)
    pg.wait_for_timeout(2000)
    pg.screenshot(path="/tmp/dash.png")
    b.close()
```

### Viewer Streamlit

```bash
streamlit run src/show_colect_data/viewer_patches_npy.py \
    --server.port 8052 --server.headless true
# apontar base_dir para pasta com estrutura <region>/<year>/patch_*.npy
```

---

## Run (human path — acesso remoto)

```bash
# no servidor
python3 src/dash/dashboard_predict.py --index-file ~/dados/predict_index.json --port 8051

# no notebook local
ssh -L 8051:localhost:8051 user@servidor
# abrir http://localhost:8051
```

---

## Verificar imagens faltantes

```bash
python3 src/posprocessing/check_missing_tifs.py \
    --img-dir /srv/almacen/db_images/dataset_fotovoltaica_TIFreg
```

Saída: lista de `{region_id}_{ano}` ausentes agrupados por região.

---

## Gotchas

- **`predict_index.json` não existe** → `makePredict_fromTIF.py` precisa ter rodado ao menos uma vez. Alternativa: criar manualmente (ver seção Dados de teste).
- **`files().delete(body={'trashed':True})` no Drive** → API errada; `delete` ignora o body e faz deleção permanente. O script correto usa `files().update(fileId=..., body={'trashed':True})`.
- **Conta de serviço sem permissão de escrita no Drive** → o download funciona normalmente; só o step de mover para lixeira falha com 403 — tratado com try/except separado em `read_TIF_from_gdrive.py`.
- **`pip install` bloqueado no Arch Linux** → adicionar `--break-system-packages`.
- **Dash 4.x** — callback `Output`/`Input` vêm de `dash` diretamente (não `dash.dependencies`).
- **Bandas do mosaico**: ordem rasterio é 1=blue, 2=green, 3=red, 4=pvi, 5=pvpi → RGB usa `src.read([3, 2, 1])`.

---

## Troubleshooting

| Sintoma | Causa | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: No module named 'dash'` | dependências não instaladas | `pip install dash dash-bootstrap-components rasterio scipy pillow --break-system-packages` |
| `FileNotFoundError: predict_index.json não encontrado` | index inexistente | rodar `makePredict_fromTIF.py` ou criar manualmente |
| Dashboard abre mas imagens em branco | `dd-year` callback ainda não disparou | aguardar 2 s ou selecionar região/ano manualmente |
| `HttpError 403` ao mover arquivo para lixeira | conta de serviço só tem leitura | comportamento esperado; o download foi bem-sucedido |
