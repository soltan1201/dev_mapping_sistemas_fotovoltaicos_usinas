# Números-chave e glossário

Use esta tabela para citar números com segurança em qualquer artigo — se um número que você
precisa não está aqui nem em `01-projeto-e-pipeline.md` / `02-arquiteturas-modelos.md`, não o
invente: pergunte a quem pediu o artigo ou marque como "não informado" no texto.

## Tabela de números-chave

| Métrica | Valor | Observação |
|---|---|---|
| Regiões candidatas gerais mapeadas | ~91 | Além do subconjunto rural dedicado |
| Cobertura temporal | 2016–2025 (10 anos) | |
| Combinações região × ano | 910 | 91 regiões × 10 anos |
| Resolução da imagem-fonte (Planet NICFI) | 4,77 m/pixel | |
| Resolução do asset público final publicado | 10 m/pixel | Padronizado com o restante da coleção MapBiomas |
| Tamanho do patch de treino/inferência | 256×256 pixels | |
| Bandas espectrais de entrada (versão atual) | 5 | 3 de cor (blue, green, red) + 2 índices derivados (pvi, pvpi) |
| Bandas espectrais de entrada (versão legado) | 8 | 4 de cor/NIR + 4 índices derivados |
| Épocas de treino (notebook atual) | 250 | |
| Proporção negativo:positivo antes do balanceamento | ≈ 5,7:1 | ~15% dos patches de treino eram positivos |
| Proporção negativo:positivo após balanceamento | 3:1 | Fragmentos 100% negativos ficam em quarentena |
| Cabeças de segmentação implementadas | 5 | U-Net, DeepLabV3+, PSPNet, FPN, SegFormer-B5 |
| Backbones implementados | 11 chaves / 10 arquiteturas distintas | ResNeXt50 é, na prática, um alias de ResNet50 (ver `02-arquiteturas-modelos.md`) |
| Melhor val_dice registrado (geração legado, 8 bandas) | 0,910 | Backbone ResNet-152 |
| Melhor val_iou registrado (geração legado, 8 bandas) | 0,920 | Backbone ResNet-152 |

## Linha do tempo simplificada de uma predição, do dado bruto ao mapa público

1. Imagem de satélite (Planet NICFI, via Google Earth Engine)
2. Exportação para Google Drive (formato TFRecord ou GeoTIFF)
3. Download para servidor local
4. Predição pelo modelo de deep learning (segmentação semântica, pixel a pixel)
5. Conversão para Cloud Optimized GeoTIFF (COG)
6. Upload para Google Cloud Storage
7. Ingestão de volta no Google Earth Engine como asset de coleção de imagens
8. Curadoria/controle de qualidade por região (regras manuais)
9. Mosaico nacional final por ano → asset público da coleção MapBiomas
10. Cálculo de área por bioma/ano → gráficos e página pública de divulgação

## Glossário

**MapBiomas** — iniciativa brasileira de mapeamento anual de uso e cobertura da terra por
satélite, organizada em coleções temáticas (ex.: agropecuária, água, energias renováveis).

**Google Earth Engine (GEE)** — plataforma de processamento geoespacial em nuvem do Google,
usada para acessar imagens de satélite e rodar análises em escala planetária sem precisar
baixar os dados brutos manualmente.

**Planet NICFI** — programa que disponibiliza gratuitamente mosaicos de satélite de altíssima
resolução (Planet Labs) para florestas tropicais, financiado pela iniciativa norueguesa
Norway's International Climate & Forest Initiative.

**Segmentação semântica** — tarefa de deep learning que classifica cada pixel de uma imagem em
uma categoria (aqui: "painel solar" ou "não é painel solar"), produzindo uma máscara no formato
da imagem original — diferente de classificação de imagem (que dá um rótulo único para a
imagem inteira) ou detecção de objetos (que dá caixas delimitadoras).

**Backbone** — rede de classificação de imagens já consolidada na literatura (ex.: ResNet,
Inception), reaproveitada como extrator de features dentro de um modelo maior de segmentação.

**Transfer learning** — reaproveitar um modelo treinado em uma tarefa/dataset para acelerar o
aprendizado em outra tarefa. Neste projeto, apenas a **topologia** dos backbones é reaproveitada
— os pesos não, porque a entrada tem 5 bandas espectrais em vez das 3 (RGB) do ImageNet
original (ver `02-arquiteturas-modelos.md`).

**Patch** — um recorte pequeno e de tamanho fixo de uma imagem maior (aqui, 256×256 pixels),
usado como unidade de entrada do modelo durante treino e inferência.

**TFRecord** — formato de arquivo binário do TensorFlow, eficiente para armazenar grandes
volumes de dados de treino/inferência.

**GeoTIFF** — formato de imagem TIFF com metadados de georreferenciamento (coordenadas
geográficas embutidas), padrão em sensoriamento remoto e SIG (sistemas de informação
geográfica).

**COG (Cloud Optimized GeoTIFF)** — variante do GeoTIFF organizada internamente para permitir
leitura eficiente de apenas partes do arquivo remotamente, sem baixá-lo por inteiro.

**Bioma** — uma das grandes regiões ecológicas do Brasil (ex.: Amazônia, Cerrado, Caatinga,
Mata Atlântica, Pampa, Pantanal), unidade geográfica usada nas análises de área por região
deste projeto.

**Dice / IoU (Intersection over Union)** — métricas padrão para avaliar a qualidade de uma
segmentação, comparando a máscara prevista pelo modelo com a máscara real (rótulo). Variam de 0
a 1; quanto mais próximo de 1, melhor a sobreposição entre predição e realidade.

**Loss function (função de perda)** — a métrica que o treino de uma rede neural tenta minimizar
a cada passo. Este projeto usa uma combinação de Focal Tversky (adequada quando uma classe é
muito mais rara que a outra, como painéis solares em relação ao resto da paisagem) com uma
componente de borda que penaliza especificamente erros no contorno da máscara prevista.

**Weights=None** — na biblioteca usada neste projeto, instanciar uma arquitetura sem carregar
pesos pré-treinados, ou seja, com pesos aleatórios que serão inteiramente aprendidos durante o
treino deste projeto específico.

**"L5" / "L9"** — nomenclatura interna do projeto para as duas gerações do modelo: L5 = versão
atual, 5 bandas espectrais de entrada; L9 = versão anterior/legado, 8 bandas espectrais de
entrada (o "9" no nome não corresponde ao número exato de bandas — é apenas um rótulo interno).

## Política de honestidade (lembrete)

Todo número técnico usado em um artigo deve vir desta tabela, de `01-projeto-e-pipeline.md` ou
de `02-arquiteturas-modelos.md`, ou ter sido explicitamente confirmado por quem pediu o artigo.
Não interpole, arredonde de forma enganosa, ou complete lacunas com estimativas plausíveis
apresentadas como fato.