# Notebooks — Pós-processamento das classificações fotovoltaicas

Fluxo de trabalho após a inferência do modelo: **diagnosticar falhas → compilar/reduzir
os resultados → inspecionar visualmente**. Todos rodam no **Google Colab** com o Drive
montado em `/content/drive`.

## Visão geral do fluxo

```
makePredict_fromTIF_sortedByQuantity_colab.ipynb   (inferência — gera os *_pred.tif)
                    │
                    ├─────────────────────────────┐
                    ▼                              ▼
  ┌───────────────────────────────┐   ┌────────────────────────────────────┐
  │ 1. puxar_visualizar_falhas     │   │ 2. compilar_reduzir_classificadas   │
  │    diagnostica / apaga TIFs    │   │    imagem | predict + versão        │
  │    de entrada corrompidos      │   │    reduzida (só onde há FV)         │
  └───────────────────────────────┘   └────────────────────────────────────┘
                                                    │  (gera *_reduzido.tif)
                                                    ▼
                                  ┌────────────────────────────────────┐
                                  │ 3. visualizar_aleatorio_reduzida    │
                                  │    inspeciona reduzidas (aleatório  │
                                  │    ou por nome)                     │
                                  └────────────────────────────────────┘
```

## Pastas no Drive (padrão)

| Variável       | Caminho padrão                                                 | Conteúdo                          |
|----------------|----------------------------------------------------------------|-----------------------------------|
| `INPUT_DIR`    | `.../DS_FV_TIFs_scaled/PREDICT_V2`                             | TIFs de entrada (5 bandas)        |
| `OUTPUT_DIR`   | `.../DL_fotovoltaica/tif_classificadas_2025`                  | predições `*_pred.tif`            |
| `REDUCED_DIR`  | `.../DL_fotovoltaica/tif_reduzidas_2025`                      | reduzidas `*_img/_pred_reduzido`  |

Ajuste os caminhos na célula **PARÂMETROS** de cada notebook se necessário.

---

## 1. `puxar_visualizar_falhas_colab.ipynb`

**Quando usar:** depois de rodar o predict, se alguma imagem não gerou `_pred.tif`.

As falhas observadas **não são do modelo**, e sim de **integridade do arquivo no Drive**
(download parcial vindo do GEE):

- `corrompido (header inválido)` — `not recognized as being in a supported file format`
- `truncado (download incompleto)` — `TIFFReadEncodedTile() failed ... got N, expected M`

Esses arquivos **precisam ser re-exportados do GEE** — puxar de novo não conserta.

**O que faz:**
1. Detecta automaticamente os TIFs de `INPUT_DIR` sem `_pred.tif` (ou usa `NOMES_MANUAIS`).
2. Diagnostica a integridade e salva `diagnostico_falhas.csv`.
3. Copia os arquivos para uma pasta local do Colab (`FALHAS_DIR`) para inspeção.
4. Visualiza (RGB) os que abrem; marca os corrompidos para re-exportação.
5. **Célula 8 — apaga** os TIFs com falha do Drive para poder re-exportar.
   Protegida por `CONFIRMAR_APAGAR` (padrão `False` = só simula) e por `STATUS_APAGAVEIS`
   (nunca apaga um arquivo `OK`).

## 2. `compilar_reduzir_classificadas_colab.ipynb`

**Quando usar:** para compilar os resultados e gerar um dataset enxuto, guardando só
as regiões com fotovoltaica.

**O que faz:**
1. Pareia cada original ↔ `_pred.tif`.
2. `visualizar_par()` — mostra **original (RGB) | predict** lado a lado.
3. `reduzir_por_patches()` — fatia original e predict no mesmo grid de `PATCH_SIZE`,
   mantém só os pedaços cujo **predict tem algum pixel = 1**, descarta linhas/colunas
   vazias do grid e **remonta** os sobreviventes numa imagem compacta (original e predict
   alinhados).
4. Salva `<id>_img_reduzido.tif` e `<id>_pred_reduzido.tif` em `REDUCED_DIR`.

> ⚠ O geotransform da imagem reduzida é **aproximado** — remover as lacunas quebra a
> continuidade geoespacial. Serve para inspeção/dataset, não para medir coordenadas.

## 3. `visualizar_aleatorio_reduzida_colab.ipynb`

**Quando usar:** para conferir rapidamente as reduzidas geradas no passo 2.

**O que faz:**
- **Célula 4** — sorteia uma reduzida (`random.choice`) e mostra **original reduzido (RGB) |
  predict reduzido**. Reexecute para sortear outra. `SEED` fixa o sorteio se quiser reprodutível.
- **Célula 5** — escolhe por **nome específico** (`NOME_ESCOLHIDO`), aceitando o identificador
  com ou sem sufixo. Se não encontrar, lista os stems disponíveis.

---

## Ordem recomendada

1. `makePredict_fromTIF_sortedByQuantity_colab.ipynb` — roda a inferência.
2. `puxar_visualizar_falhas_colab.ipynb` — resolve as imagens que falharam (diagnostica /
   apaga → re-exporta do GEE → roda o predict de novo só nelas).
3. `compilar_reduzir_classificadas_colab.ipynb` — gera as versões reduzidas.
4. `visualizar_aleatorio_reduzida_colab.ipynb` — inspeciona o resultado.
