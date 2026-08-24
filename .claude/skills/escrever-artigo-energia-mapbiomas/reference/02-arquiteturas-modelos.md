# Arquiteturas de deep learning usadas no projeto

Este documento descreve as arquiteturas de rede neural realmente usadas no projeto de
mapeamento de sistemas fotovoltaicos, de forma independente do código-fonte. Use-o para
qualquer artigo com foco técnico/ML. Todos os fatos aqui (ano, autores, ideia central, "por que
esse nome") foram verificados contra os papers originais — os diagramas e explicações são
descrições próprias, não reprodução das figuras originais.

## Conceito central: cabeça de segmentação × backbone

O modelo final é montado a partir de duas peças combináveis:

- **Cabeça de segmentação:** a topologia que efetivamente produz o mapa de pixels
  (painel solar / não é painel solar). É a peça "principal" do modelo.
- **Backbone:** um encoder de classificação de imagens já consolidado na literatura
  (ResNet, Inception, EfficientNet etc.), plugado dentro da cabeça de segmentação para
  extrair features em múltiplas escalas, em vez de a cabeça construir seu próprio
  extrator de features do zero.

Quatro das cinco cabeças de segmentação implementadas (U-Net, DeepLabV3+, PSPNet, FPN) aceitam
qualquer um dos backbones como encoder opcional. A quinta (SegFormer-B5) é uma exceção: é um
bloco Transformer autocontido que não aceita nenhum dos backbones desta lista.

## Por que não há transfer learning de pesos (somente de topologia)

Este é o ponto mais importante para qualquer leitor com alguma familiaridade em deep learning,
e frequentemente motivo de confusão: **os backbones aqui NÃO usam os pesos pré-treinados no
ImageNet.** A entrada real do modelo é uma imagem de 5 bandas espectrais (blue, green, red, e
dois índices derivados — ver `01-projeto-e-pipeline.md`), incompatível com os pesos de um
backbone treinado originalmente para 3 canais RGB. Por isso, todo backbone é instanciado "do
zero" (sem pesos pré-carregados) — **reaproveita-se apenas a topologia** (a sequência e o
formato das camadas) de cada arquitetura consagrada, e todos os pesos são aprendidos
inteiramente a partir dos dados deste projeto durante o treino.

Isso é diferente do uso mais comum dessas arquiteturas na literatura (fine-tuning a partir de
pesos ImageNet) e deve ser deixado claro em qualquer artigo técnico, para não sugerir que o
projeto se beneficia do conhecimento visual genérico acumulado nesses pesos.

---

## 1. Cabeças de segmentação

### U-Net (com variante Focus U-Net — a usada atualmente)

- **Origem:** Ronneberger, Fischer & Brox, 2015, MICCAI (arXiv:1505.04597).
- **Contexto original:** segmentação de imagens de microscopia biomédica, com poucos exemplos
  de treino disponíveis.
- **Ideia central:** um encoder-decoder simétrico — um caminho de contração (downsampling
  sucessivo, captura contexto cada vez mais abstrato) e um caminho de expansão (upsampling
  sucessivo, recupera resolução espacial), ligados por conexões diretas ("skip connections")
  que copiam os mapas de features de alta resolução do encoder direto para o nível equivalente
  do decoder. Isso combina semântica profunda (o quê) com detalhe espacial fino (onde), mesmo
  com poucos dados de treino.
- **Por que o nome "U-Net":** o caminho de contração desce visualmente à esquerda até um ponto
  mais estreito (bottleneck) e o caminho de expansão sobe à direita, ligados por conexões
  horizontais — o diagrama resultante forma a letra "U".
- **Uso neste projeto:** é a cabeça configurada no treino atual. A variante usada,
  **Focus U-Net**, substitui a simples concatenação das skip connections por um "Focus Gate" —
  um mecanismo de atenção que combina atenção de canal e espacial com um filtro focal, ajudando
  o modelo a dar mais peso às regiões/canais mais relevantes para distinguir painel solar do
  fundo, em vez de tratar toda a feature copiada com peso igual.

### DeepLabV3+

- **Origem:** Chen, Zhu, Papandreou, Schroff & Adam, 2018, ECCV (arXiv:1802.02611).
- **Resultados reportados no paper original:** 89,0% mIoU no PASCAL VOC 2012 · 82,1% mIoU no
  Cityscapes.
- **Ideia central:** combina o módulo **ASPP** (Atrous Spatial Pyramid Pooling) — convoluções
  "atrous"/dilatadas em paralelo, com taxas de dilatação diferentes, que capturam contexto em
  múltiplas escalas sem reduzir a resolução espacial — com um decoder leve que recupera bordas
  de objetos fundindo de volta uma feature de baixo nível do encoder.
- **Por que o nome:** é a 4ª geração da série DeepLab (v1: convolução atrous + refinamento por
  campo aleatório condicional; v2: introduz o ASPP; v3: ASPP aprimorado); o "+" indica
  justamente o módulo decoder adicionado sobre o encoder do DeepLabv3.
- **Uso neste projeto:** implementada com suporte opcional a qualquer um dos backbones desta
  lista como encoder do ASPP, no lugar do encoder original do paper.

### PSPNet

- **Origem:** Zhao, Shi, Qi, Wang & Jia, 2017, CVPR (arXiv:1612.01105).
- **Resultados reportados no paper original:** 85,4% mIoU (VOC12) · 80,2% mIoU (Cityscapes) ·
  1º lugar no desafio de scene parsing do ILSVRC 2016.
- **Ideia central:** o **Pyramid Pooling Module (PPM)** agrega contexto em múltiplas escalas
  espaciais em paralelo (de uma única região global até uma grade fina 6×6) e funde esse
  contexto de volta com as features locais — corrigindo um problema comum de redes totalmente
  convolucionais simples, que confundem categorias parecidas por não "enxergarem" o contexto
  global da cena.
- **Por que o nome "PSPNet":** Pyramid Scene Parsing Network — nomeado diretamente pelo módulo
  de pooling em pirâmide que "interpreta" (parses) a cena inteira em múltiplas escalas.
- **Uso neste projeto:** disponível como cabeça vanilla ou com qualquer um dos backbones desta
  lista como extrator de features antes do módulo de pooling piramidal.

### FPN — Feature Pyramid Network

- **Origem:** Lin, Dollár, Girshick, He, Hariharan & Belongie, 2017, CVPR (arXiv:1612.03144).
- **Contexto original:** originalmente proposta para **detecção de objetos** (dentro do
  Faster R-CNN), não para segmentação — vale mencionar essa origem se o artigo discutir a
  arquitetura em detalhe, para não sugerir que foi criada com este propósito.
- **Ideia central:** aproveita a hierarquia piramidal natural de uma rede convolucional,
  acrescentando um caminho "top-down" com conexões laterais que funde features de alta
  resolução mas semântica fraca (vindas do caminho bottom-up/encoder) com features de baixa
  resolução mas semântica forte (vindas do topo da rede) — produzindo uma pirâmide completa de
  mapas de features, todos semanticamente fortes, com custo computacional extra pequeno.
- **Por que o nome "FPN":** descreve literalmente sua saída — uma pirâmide de mapas de features
  em múltiplas resoluções, análoga às pirâmides de imagem clássicas do processamento de imagem,
  mas construída a partir de features aprendidas por uma rede neural.
- **Uso neste projeto:** disponível como opção de cabeça, com ou sem backbone — mas não é a
  cabeça selecionada no treino atual (que usa U-Net).

### SegFormer-B5 (nova, ainda sem treinamento registrado)

- **Origem:** Xie, Wang, Yu, Anandkumar, Alvarez & Luo, 2021, NeurIPS (arXiv:2105.15203).
- **Ideia central:** um encoder **Transformer** hierárquico ("Mixed Transformer", MiT) que
  produz features multi-escala em 4 estágios (como uma rede convolucional tradicional faria),
  mas usando self-attention eficiente em vez de convoluções — e, diferente de propostas
  anteriores de segmentação por transformer, **sem positional encoding**, o que evita o
  problema de precisar reinterpolar posições ao mudar a resolução de teste. O decoder é
  deliberadamente simples: um módulo "All-MLP" que projeta os 4 estágios para o mesmo número de
  canais, faz upsample, concatena tudo e funde com duas camadas totalmente conectadas — funciona
  bem porque o encoder Transformer já dá um campo receptivo grande em qualquer estágio,
  dispensando um decoder pesado do tipo usado em U-Net ou PSPNet.
- **Por que o nome "SegFormer":** contração de "Segmentation" + "Transformer" — um design
  deliberadamente simples e eficiente, em contraste com propostas anteriores de segmentação por
  transformer que dependiam de positional encoding e decoders mais pesados.
- **Uso neste projeto:** é a adição mais recente ao conjunto de cabeças disponíveis. Ao
  contrário das outras quatro, não aceita nenhum dos backbones desta lista como encoder (é um
  bloco Transformer autocontido) e tem aproximadamente 82 milhões de parâmetros. **Importante
  para qualquer artigo:** está disponível como opção no código, mas até o momento **não tem
  nenhuma execução de treino registrada** — não deve ser apresentada como um resultado do
  projeto, apenas como uma capacidade adicionada e ainda não validada experimentalmente.

---

## 2. Backbones (encoders de classificação)

Onze combinações de nome de backbone foram implementadas (dez arquiteturas distintas — ver
nota sobre ResNeXt50 abaixo). Para cada uma: tamanho de entrada e número de classes são os
originais do paper (treino em ImageNet, 3 canais RGB) — não o que roda de fato neste projeto,
que sempre usa (256, 256, 5) e nenhum peso pré-treinado (ver seção acima).

### InceptionV3

- **Origem:** Szegedy, Vanhoucke, Ioffe, Shlens & Wojna, 2016, CVPR (arXiv:1512.00567).
- **Entrada original:** 299×299×3 · **Parâmetros:** ≈ 23,9 milhões.
- **Ideia central:** módulos "Inception" rodam ramos de convolução paralelos (1×1, 3×3,
  pooling) que são concatenados — uma rede "mais larga" em vez de apenas mais profunda. Esta
  versão específica introduz a fatoração de convoluções grandes em convoluções menores mais
  baratas (ex.: uma 5×5 vira duas 3×3 empilhadas), entre outras otimizações de eficiência.
- **Por que o nome "Inception":** vem do paper original do GoogLeNet (2014), cujos autores
  citam em nota de rodapé o meme "we need to go deeper" — referência ao filme "A Origem"
  ("Inception", 2010) — como brincadeira sobre a profundidade da rede; o nome também reflete o
  design multi-ramo, "mais largo", característico da família.
- **Status de treino neste projeto:** sem nenhuma execução de treino registrada até o momento.

### ResNet-50, ResNet-101, ResNet-152

- **Origem (as três):** He, Zhang, Ren & Sun, 2016, CVPR (arXiv:1512.03385).
- **Entrada original:** 224×224×3 em todas · **Parâmetros:** ≈ 25,6 M (50) · ≈ 44,5 M (101) ·
  ≈ 60,2 M (152).
- **Ideia central:** resolve o "problema de degradação" observado em redes muito profundas —
  a partir de certa profundidade, o erro de *treino* (não apenas de teste) piora ao empilhar
  mais camadas, o que não é explicado por overfitting. A solução: em vez de uma camada aprender
  diretamente a função desejada H(x), ela aprende a função residual F(x) = H(x) − x, somada de
  volta a x por um atalho de identidade — uma reformulação que facilita muito a otimização de
  redes profundas, porque o atalho fornece uma solução trivial de "não fazer nada" caso a
  camada extra não ajude.
- **Diferença entre as três variantes:** apenas o número de blocos residuais empilhados em cada
  estágio da rede (mais blocos = mais profundidade = mais parâmetros); a ideia central e o tipo
  de bloco são idênticos nas três.
- **Por que o nome "Residual":** vem diretamente da função residual F(x) = H(x) − x que cada
  bloco aprende, em vez do mapeamento direto H(x).
- **Status de treino neste projeto:** na geração anterior do modelo (8 bandas), o ResNet-152
  foi o melhor resultado obtido entre todos os backbones testados; o ResNet-50 teve uma
  execução parcial razoável e uma que não convergiu; o ResNet-101 não convergiu. Na geração
  atual (5 bandas), há checkpoints salvos para resnet152 e resnet50, mas ainda sem métricas
  finais registradas.

### ResNeXt-50 (32×4d) — atenção: nome não corresponde à implementação atual

- **Origem:** Xie, Girshick, Dollár, Tu & He, 2017, CVPR (arXiv:1611.05431).
- **Entrada original:** 224×224×3 · **Parâmetros (paper):** ≈ 25,0 M — deliberadamente
  comparável ao ResNet-50 (25,6 M), para isolar o efeito do design proposto.
- **Ideia central (da arquitetura original):** introduz a **cardinalidade** — o número de
  caminhos de transformação paralelos e idênticos dentro de um bloco — como uma terceira
  dimensão de design de rede, além de profundidade e largura. Cada bloco divide a entrada em 32
  caminhos de baixa dimensão, aplica a mesma transformação (não customizada à mão, como no
  Inception) a cada um, agrega o resultado e só então soma o atalho residual, no mesmo espírito
  do ResNet. Implementado de forma eficiente como convolução agrupada.
- **Por que o nome "ResNeXt":** o próprio paper descreve o nome como sugerindo "a próxima
  dimensão" (*next dimension*) — a cardinalidade como um novo eixo de design após profundidade
  e largura — mantendo a herança "Res" (residual) do ResNet. É uma afirmação direta dos
  autores, não uma interpretação externa.
- **Ressalva importante para qualquer artigo que mencione esta peça:** na implementação usada
  neste projeto, o nome "resnext50" na prática constrói uma rede **estruturalmente idêntica ao
  ResNet-50** (mesmo bloco Bottleneck, sem as convoluções agrupadas / cardinalidade real
  descritas acima) — um efeito de uma biblioteca de terceiros usada no projeto não oferecer uma
  implementação nativa de ResNeXt. **Não descreva "resnext50" neste projeto como se fosse a
  arquitetura ResNeXt original** — se for citada, deixe claro que, tecnicamente, o resultado
  obtido sob esse nome equivale a um ResNet-50. Note que, apesar disso, essa configuração teve
  bons resultados na geração anterior do modelo (segunda melhor entre os backbones testados) —
  o que só reforça que a comparação de resultados deve ser lida como "ResNet-50 vs. os demais",
  não como uma validação da arquitetura ResNeXt de fato.

### MobileNet (v1)

- **Origem:** Howard et al., 2017, Google (arXiv:1704.04861).
- **Entrada original:** 224×224×3 · **Parâmetros:** ≈ 4,3 M · Acurácia Top-1 original: 70,4%.
- **Ideia central:** substitui convoluções padrão por **convoluções separáveis em
  profundidade** (depthwise separable): uma convolução depthwise (um filtro por canal, só
  espacial) seguida de uma convolução pointwise 1×1 (mistura os canais) — reduz computação e
  parâmetros em cerca de 8-9 vezes com acurácia comparável. Dois hiperparâmetros globais
  permitem trocar acurácia por velocidade: um multiplicador de largura (afina o número de
  canais) e um multiplicador de resolução (reduz o tamanho da imagem de entrada).
- **Por que o nome "MobileNet":** foi desenhada explicitamente para rodar em celulares e
  dispositivos embarcados — eficiência e baixa latência foram a prioridade de design desde o
  início, não a maximização de acurácia a qualquer custo.
- **Status de treino neste projeto:** na geração anterior (8 bandas), convergiu bem, com
  resultado competitivo entre os backbones leves testados.

### Xception

- **Origem:** Chollet, 2017, CVPR (arXiv:1610.02357).
- **Entrada original:** 299×299×3 · **Parâmetros:** ≈ 22,9 M.
- **Ideia central:** reinterpreta os módulos Inception como um meio-termo entre convolução
  padrão e a separação total entre correlações espaciais e correlações entre canais. Esta
  arquitetura leva essa ideia ao limite: cada módulo Inception é substituído por uma convolução
  separável em profundidade, desacoplando completamente o aprendizado espacial do aprendizado
  entre canais; conexões residuais (no estilo ResNet) são usadas ao longo de toda a rede.
- **Por que o nome "Xception":** contração de "Extreme Inception" — leva a hipótese central do
  Inception (de que correlações espaciais e entre canais podem ser tratadas separadamente) à
  sua conclusão extrema.
- **Status de treino neste projeto:** na geração anterior (8 bandas), convergiu bem (bom Dice,
  IoU um pouco mais baixo que os melhores da série). É também o backbone usado originalmente no
  paper do DeepLabV3+, uma das cabeças de segmentação desta lista.

### EfficientNet-B0, B3 e B7

- **Origem (as três):** Tan & Le, 2019, ICML (arXiv:1905.11946).
- **Entrada original:** 224×224×3 (B0) · 300×300×3 (B3) · 600×600×3 (B7).
- **Parâmetros / acurácia Top-1 original:** ≈ 5,3 M / 77,1% (B0) · ≈ 12,3 M / 81,6% (B3) ·
  ≈ 66,7 M / 84,3% (B7).
- **Ideia central:** em vez de escalar profundidade, largura ou resolução de entrada de forma
  isolada e arbitrária (prática comum antes deste paper), introduz o **escalonamento
  composto**: um único coeficiente escala as três dimensões simultaneamente, segundo fórmulas
  fixas. A rede-base (B0) foi encontrada por busca de arquitetura neural (NAS) e é construída
  com blocos **MBConv** (bottleneck invertido móvel, herdado de arquiteturas anteriores da
  família MobileNet/MnasNet) com mecanismo de squeeze-and-excitation. B1 a B7 são a mesma
  arquitetura-base escalada para coeficientes crescentes — mais profundidade, largura e
  resolução, mais custo computacional, mais acurácia.
- **Por que o nome "EfficientNet":** o método de escalonamento composto atinge muito mais
  acurácia por unidade de custo computacional do que o escalonamento arbitrário praticado
  anteriormente; o número B0-B7 indexa o valor do coeficiente de escala usado.
- **Status de treino neste projeto:** B0 e B3 sem nenhuma execução de treino registrada; B7 tem
  um checkpoint salvo na geração atual (5 bandas), sem métricas finais registradas ainda, e é o
  backbone citado no notebook usado para gerar predições de exemplo.

### Inception-ResNet-v2

- **Origem:** Szegedy, Ioffe, Vanhoucke & Alemi, 2016/2017, AAAI (arXiv:1602.07261).
- **Entrada original:** 299×299×3 · **Parâmetros:** ≈ 55,9 M.
- **Ideia central:** um híbrido que combina módulos Inception (multi-ramo) com conexões
  residuais/atalho ao redor de cada bloco Inception — acelera bastante a convergência do treino
  em relação ao Inception puro, com custo computacional parecido, sem precisar do pooling caro
  de concatenação que versões anteriores do Inception usavam nas etapas de redução de
  resolução.
- **Por que o nome:** fusão literal de "Inception" (herança do design multi-ramo) com "ResNet"
  (conexões residuais); o "v2" distingue esta versão de uma variante "v1" menor introduzida no
  mesmo paper.
- **Uso e status neste projeto:** é o backbone configurado no treino atual, combinado com a
  cabeça Focus U-Net descrita acima — a combinação mais recente e "oficial" do pipeline no
  momento em que este material foi escrito, ainda sem métricas finais registradas (o
  checkpoint foi salvo, mas o histórico de treino/validação não).

---

## Status de treinamento consolidado

Para evitar apresentar capacidades não testadas como resultados, aqui está o resumo honesto do
que de fato tem evidência de treino, por geração do modelo:

**Geração anterior (8 bandas espectrais, "legado") — com métricas finais registradas:**
resnet152 (melhor resultado da série), resnext50 (na prática, ResNet-50 — ver ressalva acima),
xception e mobilenet convergiram bem; resnet50 teve uma execução parcial razoável e uma que não
convergiu; resnet101 não convergiu.

**Geração atual (5 bandas espectrais) — checkpoint salvo, mas sem métricas finais registradas
ainda:** efficientnetb7, inceptionresnetv2 (configuração atual do notebook de treino),
resnet152, resnext50 e resnet50.

**Sem nenhuma execução de treino registrada até o momento, em nenhuma das duas gerações:**
inceptionv3, efficientnetb0, efficientnetb3, e a cabeça de segmentação SegFormer-B5 (nova).

Qualquer artigo que discuta "quão bem o modelo funciona" deve se apoiar apenas nos números da
geração anterior (os únicos com métricas registradas), deixando claro que são de uma versão
mais antiga do modelo, com 8 bandas espectrais de entrada — não da configuração de 5 bandas
usada atualmente.