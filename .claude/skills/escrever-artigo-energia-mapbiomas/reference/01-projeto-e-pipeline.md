# O projeto: mapeamento de sistemas fotovoltaicos por deep learning (MapBiomas)

Este documento descreve, de forma independente do código-fonte, o que o projeto faz e como o
pipeline funciona de ponta a ponta. Serve como base factual para artigos sobre o tema — leia
por completo antes de escrever qualquer coisa sobre o pipeline, os dados ou a escala do projeto.

## O que é o projeto

A MapBiomas — iniciativa brasileira de mapeamento anual de uso e cobertura da terra por
satélite — mantém uma coleção temática dedicada a energias renováveis. Este projeto constrói,
dentro dessa coleção, um mapa anual (2016–2025) de onde existem sistemas fotovoltaicos (usinas
solares) no Brasil, gerado automaticamente por um modelo de deep learning de **segmentação
semântica**: uma rede neural que, a partir de uma imagem de satélite, classifica cada pixel
como "painel solar" ou "não é painel solar", produzindo uma máscara binária no formato da
própria imagem.

A motivação de fundo (não detalhada nos materiais-fonte deste projeto, mas inferível do
contexto): monitorar a expansão da energia solar no território brasileiro, por bioma e por
estado, com atualização anual e cobertura nacional — algo inviável de fazer manualmente na
escala envolvida (ver números abaixo).

Um mapeamento paralelo de energia eólica existe e seus dados de área são comparados aos da
energia solar na etapa final de análise (ver seção "Análise e divulgação"), mas a metodologia
de detecção usada para eólica **não está descrita** nestes materiais — não presuma que usa o
mesmo pipeline de deep learning aqui documentado.

## Fonte de dados: imagens de satélite

- **Constelação:** Planet NICFI Monthly — mosaicos mensais de altíssima resolução (**4,77
  metros por pixel**) disponibilizados gratuitamente para florestas tropicais através do
  programa Norway's International Climate & Forest Initiative, acessados via **Google Earth
  Engine (GEE)**, a plataforma de processamento geoespacial em nuvem do Google.
- **Cobertura temporal:** 10 anos, 2016 a 2025.
- **Cobertura espacial:** ~91 regiões candidatas gerais (polígonos ao redor de ocorrências
  conhecidas ou prováveis de usinas fotovoltaicas) + um subconjunto rural dedicado (ver seção
  própria abaixo), usando como referência espacial uma grade nacional de 35 "patches" (tiles)
  que cobre todo o Brasil.
- **Escala resultante:** 91 regiões × 10 anos = **910 combinações região×ano** — a unidade
  básica de trabalho repetida em quase todas as etapas do pipeline (coleta, inferência, upload,
  auditoria).

## Duas rotas paralelas: treinamento e inferência

O pipeline se divide logo na origem em duas rotas que compartilham a mesma fonte de dados
(GEE/NICFI) mas servem propósitos diferentes:

- **Rota de treinamento:** produz o modelo de deep learning, a partir de exemplos rotulados
  manualmente (onde já se sabe que há ou não usina solar).
- **Rota de inferência:** aplica o modelo já treinado sobre toda a área de interesse, ano a
  ano, para gerar o mapa final.

### Rota de treinamento

1. **Coleta de dados de treino.** Exporta recortes ("patches") de 256×256 pixels, com 5 canais
   espectrais (ver seção de bandas) mais uma máscara de rótulo binária, como arquivos
   TFRecord (formato de dados do TensorFlow) enviados ao Google Drive via exportação em lote do
   Earth Engine. Os rótulos usados no treino vêm de vetores de usinas confirmadas, desenhados
   manualmente por especialistas em anos anteriores.
2. **Divisão do dataset.** Os TFRecords são organizados em treino (80%), validação (10%) e
   teste (10%), garantindo que patches da mesma região nunca fiquem espalhados entre splits
   diferentes — evita vazamento de dados (o modelo "decorar" uma região específica em vez de
   aprender o padrão geral).
2b. **Balanceamento de classes.** Como usinas solares ocupam uma fração pequena da área total,
   a maioria dos patches coletados não contém nenhum pixel de painel solar. Uma etapa dedicada
   corrige esse desbalanceamento: de um conjunto de treino com só ~15% de patches positivos
   (proporção negativo:positivo de aproximadamente 5,7:1), os fragmentos 100% negativos são
   colocados em quarentena e devolvidos ao conjunto apenas o suficiente para atingir uma
   proporção de 3:1 — uma técnica comum para evitar que o modelo aprenda a simplesmente prever
   "nada aqui" na maior parte do tempo.
3. **Treinamento do modelo.** Roda em GPU (Google Colab, GPU A100) usando uma arquitetura
   **Focus U-Net** (U-Net com um mecanismo de atenção customizado — ver
   `02-arquiteturas-modelos.md` para detalhes) combinada com um backbone (encoder) de
   classificação pré-existente na literatura, plugado por transfer learning **de topologia**
   (não de pesos — ver a mesma referência para o porquê). O backbone atualmente configurado é o
   **Inception-ResNet-v2**. A função de perda combina duas componentes: Focal Tversky (boa para
   classes desbalanceadas) e uma componente de borda ("Boundary loss", que penaliza erros no
   contorno da máscara). O treino roda por 250 épocas.

### Rota de inferência

4. **Coleta de dados para inferência.** Diferente do treino, aqui não há rótulo — o objetivo é
   rodar o modelo sobre toda a área. Existem dois formatos de entrada possíveis: patches
   individuais (formato legado) ou o **mosaico completo da região em um único arquivo GeoTIFF
   multibanda** (caminho principal atual), ambos exportados do Earth Engine para o Google Drive
   e depois baixados para um servidor local.
5. **Predição.** O modelo treinado roda sobre os dados de inferência. No caminho baseado em
   mosaico completo, a predição usa uma **janela deslizante** (sliding window) com sobreposição
   entre janelas vizinhas, e os pixels sobrepostos são combinados por uma média ponderada
   ("blending") para evitar descontinuidades visíveis nas bordas das janelas.
6. **Publicação incremental.** Os mosaicos de predição por região/ano são convertidos para
   **Cloud Optimized GeoTIFF (COG)** — um formato de GeoTIFF otimizado para leitura parcial
   remota — e enviados a um bucket do Google Cloud Storage, de onde são finalmente importados
   como um asset de coleção de imagens (ImageCollection) no próprio Google Earth Engine, já
   com metadados de qual modelo/backbone gerou cada predição.

## Grade rural — um subconjunto dedicado

Além das ~91 regiões candidatas gerais, existe um fluxo paralelo dedicado a áreas rurais: em
vez de processar regiões inteiras, ele processa apenas as células da grade nacional de 35
patches que efetivamente intersectam pontos conhecidos de usinas fotovoltaicas rurais. Esse
fluxo mantém seu próprio controle de progresso porque os registros de submissão padrão podem
ficar desatualizados quando a cota de exportação do Drive se esgota ou quando uma geometria
chega nula do Earth Engine — problemas operacionais reais que motivaram essa solução dedicada,
não apenas uma escolha arbitrária de design.

## Curadoria e publicação — a coleção pública

A predição bruta de cada modelo, por si só, não é o produto final publicado. Antes de virar
parte da coleção pública, os resultados passam por uma etapa de curadoria/controle de
qualidade por região, com regras manuais aplicadas caso a caso:

- **Exclusão:** regiões sem nenhuma usina solar confirmada têm sua predição zerada.
- **Mescla:** regiões cuja predição melhora ao ser cruzada com a coleção anterior (a versão
  10 do produto MapBiomas) recuperam essa cobertura adicional.
- **Corte temporal:** anos anteriores a uma data de corte específica por região têm seus pixels
  zerados, removendo falsos positivos antigos que o modelo gerava antes de a usina realmente
  existir.

Depois dessa curadoria, os tiles corrigidos de todas as regiões são mosaicados em uma única
imagem nacional por ano e publicados como o **asset público oficial** da MapBiomas, dentro do
tema "Energias Renováveis" da coleção, cobrindo 2016–2025 a 10 metros de resolução espacial —
uma resolução mais grosseira que os 4,77 m originais da imagem-fonte, escolhida para
padronização com o restante da coleção MapBiomas.

## Análise e divulgação

A partir do asset público final, a área ocupada por sistemas fotovoltaicos é calculada por
ano e por bioma diretamente no Earth Engine, e exportada como planilha (CSV). Esses números
alimentam gráficos de acompanhamento (evolução por bioma, por estado, por país; participação
percentual de cada bioma; taxa de crescimento entre anos-marco) e uma comparação lado a lado
com os dados equivalentes de energia eólica. O resultado final é publicado como uma página
web estática simples, de acesso público, com o resumo de área por ano e por bioma.

## Ferramentas de visualização e controle de qualidade

Paralelamente ao pipeline de produção, existem três ferramentas internas de apoio: um painel
interativo que mostra lado a lado a imagem de satélite original e a predição sobreposta (por
modelo/região/ano), um visualizador simples de patches de treino/inferência para inspeção
manual, e uma ferramenta de auditoria que verifica, para cada uma das 910 combinações
região×ano, se cada estágio do pipeline (dados brutos, predição, mosaico final, upload,
ingestão) foi de fato concluído.

## Bandas espectrais de entrada (versão atual)

O modelo atual não usa as bandas de cor "cruas" da imagem sozinhas — usa 3 bandas de cor mais
2 índices espectrais derivados, calculados especificamente para ajudar a distinguir painel
solar de alvos espectralmente parecidos (água, sombra, solo exposto):

| Banda | Papel |
|---|---|
| blue, green, red | Composição de cor natural (RGB), base de qualquer distinção visual |
| pvi (índice derivado) | Separa alvos "escuros" (candidatos a painel, água ou sombra) do restante da cena |
| pvpi (índice derivado, novo nesta versão) | Dentro do grupo de alvos escuros, ajuda a diferenciar painel solar de solo/vegetação |

Uma versão anterior do modelo usava um conjunto de 8 bandas (incluindo infravermelho próximo e
três índices diferentes). Essa versão de 8 bandas é chamada de legado ao longo destes materiais
e não deve ser confundida com a versão atual de 5 bandas — os nomes internos "L5" (5 bandas,
atual) e "L9" (8 bandas, legado) aparecem lado a lado em alguns contextos porque modelos das
duas gerações convivem no mesmo catálogo de modelos treinados.

## Números-chave (resumo rápido)

- 91 regiões candidatas gerais + subconjunto rural dedicado
- 10 anos de cobertura (2016–2025)
- 910 combinações região×ano
- Resolução da imagem-fonte: 4,77 m/pixel · Resolução do asset público final: 10 m/pixel
- Patches de treino/inferência: 256×256 pixels
- 5 bandas espectrais de entrada na versão atual (3 cor + 2 índices derivados)

(Tabela completa de números e um glossário de termos técnicos estão em
`03-numeros-e-glossario.md`.)

## O que NÃO está documentado nestes materiais (limites honestos)

- A metodologia de detecção usada para o mapeamento de energia **eólica** — só se sabe que os
  dados de área existem e são comparados aos de energia solar.
- Custo do projeto, tamanho da equipe, cronograma de execução, ou qualquer informação
  institucional/administrativa.
- A origem exata de alguns dos dados brutos usados nos gráficos finais — parte deles foi gerada
  manualmente, fora de qualquer script rastreado, o que significa que não há um passo
  automatizado e auditável reproduzindo esses números a partir do zero.
- Métricas finais de desempenho (Dice, IoU) para os checkpoints mais recentes do modelo atual —
  elas existem apenas para a geração anterior (8 bandas). Ver `02-arquiteturas-modelos.md` para
  o detalhamento exato de quais backbones têm métricas e quais não têm.