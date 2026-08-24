---
name: escrever-artigo-energia-mapbiomas
description: Use esta skill ao escrever artigos, posts de blog, relatórios de divulgação, threads ou qualquer texto de comunicação sobre o mapeamento de sistemas fotovoltaicos (e eólicos) da MapBiomas por deep learning e imagens de satélite. Carrega fatos verificados dos arquivos em reference/ em vez de inventar números, arquiteturas ou resultados.
---

# Escrever artigo — Mapeamento de Energia Solar/Eólica (MapBiomas)

## Contexto

Este skill empacota o conhecimento técnico de um projeto real: o mapeamento de sistemas
fotovoltaicos (usinas solares) — e, em menor grau, eólicas, citadas apenas como comparação —
da MapBiomas. É um pipeline de ponta a ponta que vai da coleta de imagens de satélite via
Google Earth Engine até um asset público de mapa anual (2016–2025), passando por um modelo de
deep learning de segmentação semântica treinado sob medida para essa tarefa.

Você (a IA usando esta skill) provavelmente **não tem acesso ao repositório de código
original**. Tudo que você precisa para escrever com precisão está nos arquivos de `reference/`
desta mesma pasta — eles foram escritos para serem lidos de forma independente, sem depender
do código-fonte, de um dashboard ou de qualquer outro contexto externo.

Este skill é portátil: pode ser copiado como está (a pasta inteira) para o `.claude/skills/`
de outro projeto, ou entregue como arquivos soltos para qualquer outro agente/ferramenta de IA.

## Antes de escrever

1. **Leia o(s) arquivo(s) de `reference/` relevantes ao ângulo do artigo** (ver tabela abaixo)
   — por completo, não por amostragem.
2. **Confirme com quem pediu o artigo, se ainda não tiver sido dito:**
   - Público-alvo (leigo/divulgação científica, técnico/ML, formulador de política pública,
     investidor, imprensa)
   - Formato e tamanho (post de blog, thread, relatório técnico, resumo executivo, matéria)
   - Idioma (o material-fonte é em português do Brasil; não assuma inglês sem confirmar)
   - Ângulo/foco (números de área e crescimento, arquitetura de IA, engenharia do pipeline,
     ou uma visão geral introdutória)
3. **Nunca invente um fato que não esteja em `reference/`.** Se o artigo pedir um dado que não
   existe ali (ex.: custo do projeto, tamanho da equipe, cronograma exato, comparação com outro
   país), pergunte a quem pediu o artigo ou marque explicitamente como "não informado" — não
   complete a lacuna com um número plausível só para o texto fluir.

## Mapa dos arquivos de referência

| Arquivo | Quando usar |
|---|---|
| `reference/01-projeto-e-pipeline.md` | Visão geral do projeto, dados de satélite, todo o pipeline (treino, inferência, publicação, análise), números-chave |
| `reference/02-arquiteturas-modelos.md` | Foco em IA/ML: cabeças de segmentação, backbones, por que cada arquitetura tem esse nome, status de treinamento |
| `reference/03-numeros-e-glossario.md` | Tabela de números para citação rápida + glossário de termos técnicos — use como apoio em qualquer artigo, independente do ângulo |

## Regras de honestidade (não negociáveis)

- **Atual vs. legado.** O pipeline já passou por uma reestruturação. A versão atual usa 5 bandas
  espectrais ("L5"); uma versão anterior usava 8 bandas ("L9"). Não misture métricas de uma
  versão como se fossem da outra sem deixar claro qual é qual.
- **Validado vs. disponível.** Várias combinações de arquitetura estão implementadas no código
  mas nunca foram treinadas, ou foram treinadas sem métricas finais registradas (ver
  `03-numeros-e-glossario.md`). Não descreva algo como "resultado" ou "conquista" se o
  material-fonte diz "sem métricas registradas" ou "não convergiu".
- **Apelidos que não refletem a implementação.** Há um caso documentado (ResNeXt50) em que o
  nome usado no código, na prática, aponta para outra arquitetura — não a original da
  literatura. Se o artigo mencionar essa peça, preserve essa ressalva; não a omita só para
  simplificar a favor de uma história mais bonita.
- **Não fabricar procedência.** Alguns dados brutos usados nas análises finais foram gerados
  manualmente, fora de qualquer script rastreado. Se perguntado sobre a origem desses números,
  é legítimo dizer isso; não invente um pipeline de geração que não existe.
- **Mapeamento eólico é citado apenas como comparação.** Os arquivos de referência não
  descrevem a metodologia/modelo usados para mapear energia eólica — só que os dados de área já
  existem e são comparados aos de energia solar na etapa de análise. Não presuma que o mesmo
  pipeline de deep learning descrito aqui foi usado para eólica.

## Estruturas sugeridas por tipo de artigo

**A) Divulgação pública / geral**
Gancho com um número de crescimento de área (ver `03-numeros-e-glossario.md`) → explicação em
linguagem simples de "como a IA encontra usinas solares em imagens de satélite" → relevância
ambiental/de política pública → chamada para o dado público (asset/coleção MapBiomas).

**B) Deep-dive técnico (ML / geoprocessamento)**
Por que um U-Net customizado com atenção (Focus Gate) em vez de um modelo pronto → por que 5
bandas espectrais com índices customizados em vez de RGB puro → por que `weights=None` (sem
transfer learning de pesos do ImageNet) → desafios de engenharia (desbalanceamento de classe,
escala do dado) → comparação honesta entre backbones, com limitações explícitas.

**C) Estudo de caso de engenharia de dados**
O pipeline ponta a ponta como um sistema distribuído: Earth Engine → Drive → treino em GPU →
inferência em mosaico → Cloud Storage → asset público — números de escala (regiões × anos,
backbones testados, formatos de arquivo) e as decisões de design que resolveram gargalos reais
(ex.: sub-pipeline dedicado para a grade rural, fila de tasks do Earth Engine).

**D) Resumo executivo / policy brief**
Números primeiro, sem jargão técnico: quanto de área foi mapeada, em quais biomas, como isso
apoia o monitoramento de energia renovável no Brasil.

## Checklist final antes de entregar o artigo

- [ ] Todo número citado existe em `03-numeros-e-glossario.md` (ou foi confirmado por quem pediu o artigo)
- [ ] Nenhuma arquitetura "disponível mas não validada" foi apresentada como resultado consolidado
- [ ] Distinção L5 (atual) vs. L9 (legado) está clara onde relevante
- [ ] Nenhum caminho de arquivo/script interno do repositório foi citado como se o leitor pudesse acessá-lo (a menos que o artigo seja explicitamente sobre o código-fonte)
- [ ] Tom e nível técnico batem com o público-alvo confirmado