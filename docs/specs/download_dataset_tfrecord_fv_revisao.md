# Spec: bases de revisão do predict (treino e inspeção)

Parte GEE: `src/show_colect_data/visualizar_status_predict_rural.js` (seção 7)
Base de treino: `src/preprocessing/download_dataset_tfrecord_fv_revisao.py`
Base de inspeção: `src/preprocessing/download_tif_fv_omissao.py`

## Objetivo

Coletar FV pequenas e lugares onde os modelos treinados com usinas falharam.
Os revisores marcam no GEE, sobre a camada de predição rural, onde o predict
acertou, onde errou e quais painéis ele não detectou. Isso gera duas bases:

1. **Treino (TFRecord, com rótulo):** áreas de `fv_confirm` e `fv_false`.
2. **Inspeção (TIF, sem rótulo):** recortes em volta dos pontos de omissão, para
   rodar o predict de cada modelo novo e ver como ele se comporta nesses casos.

## Camadas de revisão (GEE)

Desenhadas com as Drawing Tools do Code Editor e **importadas como
FeatureCollection** (cada forma desenhada é uma feição).

| Import       | Forma     | Significado                               | Vai para          | `classe` |
|--------------|-----------|-------------------------------------------|-------------------|----------|
| `fv_confirm` | retângulo | predict correto na área (VP)              | base de treino    | 1        |
| `fv_false`   | retângulo | predict errado na área (FP)               | base de treino    | 0        |
| `incompleto` | ponto     | painel que o predict não detectou (FN)    | base de inspeção  | -1       |

A seção 7 do JS adiciona `tipo`, `classe`, `year`, `area_ha` e `versao` a cada
feição e exporta um asset de tabela por camada:

```
projects/mapbiomas-arida/energias/fv_{tipo}_{YEAR}_{VERSAO_REVISAO}
    tipo = confirm | false | omissao
```

Regras:

- Exportar a coleção **com metadados** (a que `revisaoParaFC` retorna). Os
  scripts Python não dependem disso, porque o ano e a versão vêm do nome do
  asset, mas os metadados (`tipo`, `classe`, `area_ha`) ajudam na auditoria.
- O `VERSAO_REVISAO` do JS e a versão digitada nos scripts Python precisam ser
  iguais.
- A camada `Rótulo TFRecord (preview)` do JS mostra o rótulo que a base de treino
  vai receber.

## Base de treino: `download_dataset_tfrecord_fv_revisao.py`

### Rotulagem

O rótulo sai do **mapa de predição FV rural do JS**, a camada `Predição <ano>`:

```
ImageCollection(ASSET_PREDICOES [+ ASSET_PRED_RURAL]).filter(year).mosaic() > 0
```

`ASSET_PREDICOES` e `ASSET_PRED_RURAL` no Python precisam ser iguais a
`asset_predicoes` e `ASSET_PRED_RURAL` no JS.

Ordem:

1. `label` = predição do ano (pixel > 0 → 1).
2. `fv_false` vira máscara: toda mancha de FV do predict dentro dela é falso
   positivo e é apagada (`0`). Isso vale para os patches das duas camadas
   (um patch de `fv_confirm` que encosta num `fv_false` também é limpo).
3. Depois disso os patches são amostrados dentro de cada polígono.
4. Fora dos polígonos, conforme a pergunta "Rótulo fora dos polígonos":
   `pred` mantém a predição (padrão); `zero` zera tudo fora de `fv_confirm`.

### Amostragem dos patches

Cada polígono é tratado individualmente: sem união das geometrias e sem
buffer. O patch pedido já limita a área. Os pontos são calculados no Python.

- **`fv_false`:** `N_PONTOS_FALSE` (5) pontos aleatórios dentro de cada polígono.
- **`fv_confirm`:** a área dos polígonos é estratificada em `N_CLASSES_CONFIRM`
  (6) classes por quantil. A classe 0 (os menores) gera 1 patch no centro e cada
  classe seguinte dobra: 1, 2, 4, 8, 16, 32 patches (o centro mais pontos
  aleatórios dentro do polígono).

Os pontos de todos os polígonos são embaralhados (seed = ano) e divididos em
shards de no máximo 50 patches. O dry-run mostra a tabela de classes (faixa de
área, número de polígonos e pontos por polígono).

### Parâmetros de entrada

Pedidos com `input()` ao iniciar; Enter aceita o padrão entre colchetes. No fim é
mostrado um resumo que precisa ser confirmado.

| Pergunta                             | Padrão                                    | Valores válidos            | Observação |
|--------------------------------------|-------------------------------------------|----------------------------|------------|
| Destino                              | `drive`                                   | `drive`, `local`           | ver Destinos |
| Ano                                  | `2025`                                    | inteiro                    | ano do mosaico NICFI, da predição e dos polígonos |
| Versão dos assets de revisão         | `22_11_2026`                              | texto                      | sufixo do nome dos assets |
| Tipos de polígono                    | `confirm false`                           | `confirm`, `false`         | camadas a processar |
| Rótulo fora dos polígonos            | `pred`                                    | `pred`, `zero`             | ver Rotulagem |
| Pasta local de saída (só local)      | `~/almacen/db_images/tfrecord_fv_revisao` | caminho                    | criada se não existir |
| Requisições em paralelo (só local)   | `16`                                      | inteiro                    | baixe para 8 se houver muitos retries HTTP 429 |
| Dry-run                              | `n`                                       | s / n                      | só conta polígonos, pontos e imagens de predição |
| Confirmar e continuar?               | `s`                                       | s / n                      | `n` sai sem fazer nada |

Configurações fixas (bloco CONFIGURAÇÕES): `ASSET_PREDICOES`, `ASSET_PRED_RURAL`,
`PATCH_SIZE=256`, `SCALE=4.77`, `MAX_PATCHES_FILE=50`, `N_PONTOS_FALSE=5`,
`N_CLASSES_CONFIRM=6`,
`EXPORT_DRIVE_FOLDER` e os percentis de normalização do NICFI.

### Formato de saída

Igual ao de `download_dataset_tfrecord_fotovoltaica_precomp.py`:

- bandas `blue, green, red, pvi, pvpi` + `label`, escala int16 (bandas ×10000);
- cada banda é um `float_list` de 257 × 257 (kernel 128 do
  `neighborhoodToArray`), recortado para 256 × 256 no treino;
- TFRecord comprimido em GZIP;
- nome: `tfrecord_fvrev_{tipo}_{ano}_{versao}_partNNN`.

### Destinos

- **`drive`:** tasks `Export.table.toDrive` para `DS_FV_NICFI_TFR_REVISAO`. Tasks
  com o mesmo nome já `COMPLETED` são puladas.
- **`local`:** cada patch é baixado direto para o servidor com
  `ee.data.computePixels` (EPSG:3857, 4.77 m, 257 × 257 centrado no pixel do
  ponto), pelo endpoint de alto volume e com várias threads. Sem Drive e sem
  tasks.
  - Cada shard é gravado como `*.part` e renomeado para `*.tfrecord.gz` no fim;
    shards existentes são pulados, então rodar de novo retoma o download.
  - Até 5 tentativas por requisição, com espera exponencial.
  - O progresso aparece patch a patch; patch que falha nas 5 tentativas é
    pulado.
  - O log mostra quantos patches de cada shard têm FV no rótulo.
  - Grava `pontos_{tipo}_{ano}_{versao}.csv` com shard, polígono, classe, área
    e coordenadas de cada patch.
  - Limite: até 5000 polígonos por camada (um único `getInfo`).

## Base de inspeção: `download_tif_fv_omissao.py`

Sem rótulo. Para cada ponto de `fv_omissao`:

- recorte quadrado com **~500 m para cada lado** do ponto (padrão: 1 km de lado,
  210 × 210 px), alinhado à grade de pixels;
- mesmas bandas e escala dos mosaicos rurais
  (`export_mosaicTIF_GEEtoDrive_rural.py`): `blue, green, red, pvi, pvpi`, int16,
  EPSG:3857, 4.77 m;
- baixado direto para o servidor com `ee.data.computePixels` em GeoTIFF;
- nome `omissao_{system_index}_{ano}.tif`, compatível com o padrão
  `{região}_{ano}` dos scripts de predict;
- índice `omissao_index_{ano}.csv` com `system_index, lon, lat, year, arquivo,
  existe`;
- arquivos existentes são pulados (rodar de novo retoma).

### Parâmetros de entrada

| Pergunta                                   | Padrão                             | Observação |
|--------------------------------------------|------------------------------------|------------|
| Ano do asset de omissão                    | `2025`                             | parte do nome do asset |
| Versão do asset                            | `22_11_2026`                       | parte do nome do asset |
| Anos dos mosaicos                          | ano do asset                       | vários anos separados por espaço |
| Buffer em metros para cada lado do ponto   | `500`                              | lado do recorte = 2 × buffer |
| Pasta local de saída                       | `~/almacen/db_images/tif_fv_omissao` | |
| Requisições em paralelo (workers)          | `16`                               | |
| Dry-run                                    | `n`                                | só conta os pontos |
| Confirmar e continuar?                     | `s`                                | |

### Uso depois de cada treino

```bash
python src/processClass/makePredict_fromTIF.py \
    --model-path <modelo.keras> --input-dir ~/almacen/db_images/tif_fv_omissao
```

A predição de cada modelo fica numa pasta própria
(`tif_fv_omissao_<modelo>`), então dá para comparar modelos nos mesmos pontos.

## Conta e autenticação

Só a conta solkan1201 lê o mosaico Planet NICFI. Os dois scripts Python precisam
rodar no servidor autenticados com ela. A base de treino usa o projeto
`mapbiomas-brazil` (cota/cobrança), em que essa conta tem permissão:

```bash
earthengine authenticate          # logar com solkan1201
echo solkan1201 > ~/.config/earthengine/current_user.txt
```

`get_current_account()` resolve o projeto a partir de
`src/configure_account_projects_ee.py`.

## Cuidados

- Se nenhuma imagem de predição cobrir uma camada (ex.: predição rural ainda só
  no disco local), os rótulos saem zerados; o script avisa.
- Painéis não detectados que caem dentro de `fv_false` ficam com rótulo 0 na base
  de treino; marque-os em `incompleto` para entrarem na base de inspeção.
