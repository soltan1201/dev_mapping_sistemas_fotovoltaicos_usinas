#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PASSO 2: Extração de Patches para TFRecord
Lê os pontos gerados no Passo 1 e extrai os pixels das imagens Sentinel-2.
"""

import sys
import ee
import random
import json
# Ajuste o caminho das suas libs locais
pathparent = str('/home/superuser/Dados/projAlertas/proj_alertas_ML/src')
sys.path.append(pathparent)
from configure_account_projects_ee import get_current_account

projAccount = get_current_account()
print(f"Projeto selecionado >>> {projAccount} <<<")

try:
    ee.Initialize(project=projAccount)
except Exception as e:
    print("Erro de Inicialização:", e)
    raise
# ... (suas inicializações de conta e projetos permanecem as mesmas)

# ==============================================================================
# 1. PARÂMETROS DE PRODUÇÃO
# ==============================================================================
PATCH_SIZE = 256
SCALE = 10  # Mudança para Landsat (30m)
STRIDE = 128 # 50% de sobreposição (overlap) para garantir que pegamos bordas
PATCH_METERS = PATCH_SIZE * SCALE  # 7680 metros
MAX_PATCHES_PER_SHARD = 15 # Reduzido para evitar corrupção de arquivos pesados

# ASSET_ALERTS_FOLDER = "projects/mapbiomas-caatinga-cloud04/assets/Alertas/gerados_2025/Setembro2025"
# ASSET_ALERTS_FOLDER = "projects/mapbiomas-caatinga-cloud04/assets/Alertas/gerados_2025/Outubro2025"
# ASSET_ALERTS_FOLDER = "projects/mapbiomas-caatinga-cloud04/assets/Alertas/gerados_2025/Novembro2025"
# ASSET_ALERTS_FOLDER = "projects/mapbiomas-caatinga-cloud04/assets/Alertas/gerados_2025/Dezembro2025"
# ASSET_ALERTS_FOLDER = "projects/mapbiomas-caatinga-cloud04/assets/Alertas/gerados_2026/Janeiro2026"
ASSET_ALERTS_FOLDER = "projects/mapbiomas-caatinga-cloud04/assets/Alertas/gerados_2026/Fevereiro2026"
print(" ==== READING ASSETS FEATURES ===== ", ASSET_ALERTS_FOLDER)
nyear = ASSET_ALERTS_FOLDER.split("/")[-1][-4:]
print("year ===->: ", nyear)
name_month = ASSET_ALERTS_FOLDER.split("/")[-1][: -4]
print("month actived >> ", name_month)

# Pasta no Google Drive ou Cloud Storage
EXPORT_FOLDER = f'PATCHS_S2_{ASSET_ALERTS_FOLDER.split("/")[-1][:-4]}_CAAT'
print("Folder to export ", EXPORT_FOLDER)
# sys.exit()
dictDate = {
    'Janeiro':   f"{nyear}-01-01",
    'Fevereiro': f"{nyear}-02-01",
    'Marco':     f"{nyear}-03-01",
    'Março':     f"{nyear}-03-01",
    'Abril':     f"{nyear}-04-01",
    'Maio':      f"{nyear}-05-01",
    'Junho':     f"{nyear}-06-01",
    'Julho':     f"{nyear}-07-01",
    'Agosto':    f"{nyear}-08-01",
    'Setembro':  f"{nyear}-09-01",
    'Outubro':   f"{nyear}-10-01",
    'Novembro':  f"{nyear}-11-01",
    'Dezembro':  f"{nyear}-12-01"
}

# Datas
DATA_INIC = dictDate[name_month]
print("Data de inicio >>> ", DATA_INIC)
digit_month = DATA_INIC.split("-")[1]
print(digit_month)
# sys.exit()
# Lista de bandas que seu modelo espera (Baseado no seu notebook)
ALL_BANDS = ['B4', 'B8', 'B11', 'ndfia', 'afvi', 'ui', 'awei'] # Ajuste se necessário
# 4. Empilhar Tudo (Stack)
t0_bands = [b + '_t0' for b in ALL_BANDS]
t1_bands = [b + '_t1' for b in ALL_BANDS]
# Bandas
BANDS_S2 = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']

# Grade Landsat (apenas para referência de geometria se necessário)
asset_grade = 'users/CartasSol/shapes/grade_landsat_tm_Br'
grades_fc = ee.FeatureCollection(asset_grade)
print(f"we have {grades_fc.size().getInfo()} grade no featuerew collection ")

# ==============================================================================
# 2. FUNÇÕES DE PROCESSAMENTO DE IMAGEM
# ==============================================================================
def mask_clouds_s2_csplus(img):
    csPlus = ee.Image(img.get('cs_plus'))
    isClear = csPlus.select('cs').gte(0.60)
    return img.updateMask(isClear)

def add_indices_and_scale(img):
    
    img = mask_clouds_s2_csplus(img)
    # Escala para 0-1 para cálculos
    s2_scaled = img.select(BANDS_S2).divide(10000)
    
    # Endmembers (GV, NPV, Soil, Shade, Cloud)
    endmembers = [
        [0.05, 0.09, 0.04, 0.61, 0.30, 0.10], 
        [0.14, 0.17, 0.22, 0.30, 0.55, 0.30], 
        [0.20, 0.30, 0.34, 0.58, 0.60, 0.58], 
        [0.0,  0.0,  0.0,  0.0,  0.0,  0.0 ], 
        [0.90, 0.96, 0.80, 0.78, 0.72, 0.65]
    ]
    frac = s2_scaled.unmix(endmembers, True, True).rename(['gv','npv','soil','shade','cloud'])
    
    gv_shade = frac.select('gv').divide(frac.select('shade').subtract(1).abs())
    
    # Índices (Retornando para Int16 para economizar espaço no TFRecord)
    # NDFIa: (ndfia + 1) * 10000
    ndfia = (gv_shade.subtract(frac.select('soil').add(frac.select('npv'))) 
             .divide(gv_shade.add(frac.select('npv')).add(frac.select('soil')))
             .add(1).multiply(10000).toInt16().rename('ndfia'))
    
    afviImgAY = (img.expression(
        "float((b('B8') - 0.5 * b('B12')) / (b('B8') + 0.5 * b('B12')))")
            .add(1).multiply(10000).toInt16().rename(['afvi'])  
    )


    uiImgY = (img.expression(
        "float((b('B12') - b('B8')) / (b('B12') + b('B8')))")
            .add(1).multiply(10000).toInt16().rename(['ui'])
    )

    aweiY = img.expression(
                        "float(4 * (b('B3') - b('B12')) - (0.25 * b('B8') + 2.75 * b('B11')))"
                    ).add(1).multiply(10000).toInt16().rename("awei") 

    
    # Bandas originais voltam para Int16 (escala Sentinel padrão)
    return (s2_scaled.multiply(10000).toInt16()
            .addBands([ndfia, afviImgAY, uiImgY,aweiY ])
            .copyProperties(img, ['system:time_start']))


# ==============================================================================
# 2. FUNÇÃO PARA CRIAR O STACK DE PREDIÇÃO (T0 vs T1)
# ==============================================================================

def get_prediction_stack(wrs_path_n, wrs_row_n, date_t0):
    
    date_t0 = ee.Date(date_t0)
    limitGeom = grades_fc.filter(ee.Filter.eq('ID', f"{wrs_path_n}/{wrs_row_n}")).geometry()
    # print("grad size ", limitGeom.size().getInfo())
    # sys.exit()

    # Exemplo simplificado: Coleção Landsat 8 SR
    # def preprocess(img):
    #     # Aqui você deve aplicar a mesma lógica de índices (ndfia, savi) do treino
    #     # img = add_indices(img) 
    #     return img.select(ALL_BANDS)

    # 3. Construir as Imagens (T0, T1, Label)
    # sys.exit()
    # --- T1 (Imagem Atual) ---
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") 
        .filterDate(date_t0, date_t0.advance(1, 'month')) 
        .filterBounds(limitGeom) 
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70)))
    
    csPlus = (ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
        .filterDate(date_t0, date_t0.advance(1, 'month')) 
        .filterBounds(limitGeom))
    # print(f" Carregamos {s2.size().getInfo()} images Sentinel ")
    linked = ee.ImageCollection(ee.Join.saveFirst('cs_plus').apply(
        primary=s2, secondary=csPlus,
        condition= ee.Filter.equals(leftField='system:index', rightField='system:index')
    ))
    mosaicMonth = linked.map(add_indices_and_scale).median()
    # print("know bands of image before ", mosaicMonth.bandNames().getInfo())   
    # --- T0 (Imagem Passada - 15 meses antes) ---
    d_bef = ee.Date(date_t0).advance(-2, 'year')
    s2_bef = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") 
              .filterDate(d_bef.advance(-2, 'month'), d_bef.advance(3, 'month')) 
              .filterBounds(limitGeom) 
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70)))
    
    csPlus_bef = (ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED") 
                  .filterDate(d_bef.advance(-2, 'month'), d_bef.advance(3, 'month')) 
                  .filterBounds(limitGeom))

    linked_bef = ee.ImageCollection(ee.Join.saveFirst('cs_plus').apply(
        primary=s2_bef, secondary=csPlus_bef,
        condition=ee.Filter.equals(leftField='system:index', rightField='system:index')
    ))
    mosaicBefore = linked_bef.map(add_indices_and_scale).median()# .clip(limitGeom)
    # print("know bands of image before ", ee.Image(mosaicBefore).bandNames().getInfo())    

    # Renomear para diferenciar no TFRecord: B2_t0, B2_t1, etc.
    raw_t0_named = ee.Image(mosaicBefore).select(ALL_BANDS).rename(t0_bands)
    raw_t1_named = mosaicMonth.select(ALL_BANDS).rename(t1_bands)
    
    return raw_t0_named.addBands(raw_t1_named).toInt16().clip(limitGeom)


def add_bbox_dimensions(feature):
    """
    Calcula largura e altura da Bounding Box em metros.
    Adiciona as propriedades 'width_m' e 'height_m' ao feature.
    """
    # Pega o bounds projetado em metros (EPSG:3857 é seguro para medir distâncias lineares curtas)
    # O erro de projeção é desprezível para patches pequenos.
    bounds = feature.geometry().bounds(1, 'EPSG:3857')
    coords = ee.List(ee.Geometry(bounds).coordinates()).get(0)
    
    # Coordenadas do retângulo: 
    # 0: [minx, miny], 1: [maxx, miny], 2: [maxx, maxy] ...
    p0 = ee.Geometry.Point(ee.List(coords).get(0), 'EPSG:3857')
    p1 = ee.Geometry.Point(ee.List(coords).get(1), 'EPSG:3857') # Diferença no X
    p2 = ee.Geometry.Point(ee.List(coords).get(2), 'EPSG:3857') # Diferença no Y
    
    width = p0.distance(p1)
    height = p1.distance(p2)
    
    return feature.set('width_m', width, 'height_m', height)


def prepare_points_for_export(feat):
    """
    Extrai coordenadas reais para propriedades simples. 
    Isso garante a recuperação do georreferenciamento no servidor local.
    """
    coords = feat.geometry().coordinates()
    return feat.set({
        'longitude': coords.get(0),
        'latitude': coords.get(1),
        'shard_index': random.randint(0, 1000) # Para sharding
    })

# Função de Grid (Mesma lógica anterior, mas agora aplicada corretamente aos finos)
def create_grid_points_optimized(feature):
    """
        CORREÇÃO DE MEMÓRIA: Usa image.sample() ao invés de reduceToVectors.
    """
    geom = feature.geometry()
    proj = ee.Projection('EPSG:3857').atScale(SCALE)
    
    # Cria imagem de coordenadas
    coords = ee.Image.pixelCoordinates(proj)
    mask = coords.select('x').mod(STRIDE).eq(0).And(
            coords.select('y').mod(STRIDE).eq(0))
    
    # Clip pela geometria ORIGINAL (irregular)
    grid_mask = mask.clip(geom)
    
    # sample() retorna FeatureCollection de pontos diretamente dos pixels ativos
    points = grid_mask.updateMask(grid_mask).sample(
        region= geom,
        scale= SCALE,
        projection= proj,
        geometries= True,
        dropNulls= True,
        tileScale= 8
    )
    return points

# ==============================================================================
# 3. LÓGICA DE EXPORTAÇÃO DE PATCHES (TFRECORDS)
# ==============================================================================
def export_patches_for_predict(colecaoPolyg, stack_image, wrs_name, nregion):
    """
    Transforma os pontos gerados pelo seu script em tarefas de exportação de imagens.
    """
    # print(f"Bands in stack: {ee.Image(stack_image).bandNames().getInfo()}")
    # colecaoPolyg = ee.FeatureCollection(colecaoPolyg)
    size_Alerts = colecaoPolyg.size()

    # --- Label (Ground Truth Rasterizado) ---
    # Fundo = 0, Alerta = 1
    # Importante: Usamos 'paint' nos polígonos originais de alerta
    alertas_img = ee.Image(0).byte().paint(ee.FeatureCollection(colecaoPolyg), 1).rename('label')
    
    # 2. Adicionar dimensões (Map inicial inevitável para a lógica de filtro)
    fc_box =  colecaoPolyg.map(add_bbox_dimensions)

    # ==========================================================================
    # LÓGICA DE SEPARAÇÃO: DIMENSÕES, NÃO ÁREA
    # ==========================================================================
    
    # Critério: Se QUALQUER lado for maior que o patch, ele é "Grande/Longo"
    # 3. Definir o filtro de tamanho
    # Critério: Se QUALQUER lado for maior que o patch (com margem de erro)
    PATCH_THRESHOLD = PATCH_METERS - 10
    condition_large = ee.Filter.Or(
        ee.Filter.gte('width_m', PATCH_THRESHOLD),
        ee.Filter.gte('height_m', PATCH_THRESHOLD)
    )
    # 4. Filtrar coleções (Isso é rápido no servidor)
    small_polys = ee.FeatureCollection(fc_box).filter(condition_large.Not())
    large_polys = ee.FeatureCollection(fc_box).filter(condition_large)

    # 5. Calcular pontos para os pequenos (Sempre necessário se houver algum)
    # Usamos .centroid(1) para garantir que o ponto caia dentro do polígono se possível
    points_small = small_polys.map(lambda f: f.centroid(1))

    # ==========================================================================
    # OTIMIZAÇÃO: Lógica Condicional no Servidor
    # ==========================================================================
    # Usamos ee.Algorithms.If para evitar que o .map(create_grid_points_optimized).flatten()
    # seja sequer agendado se a coleção de polígonos grandes estiver vazia.
    
    all_sample_points = ee.FeatureCollection(ee.Algorithms.If(
        # Condição: Existem polígonos grandes?
        large_polys.size().gt(0),
        # Se Sim: Calcula o grid pesado e faz o merge
        points_small.merge(large_polys.map(create_grid_points_optimized).flatten()),
        # Se Não: Retorna apenas os pontos dos pequenos (pula o cálculo do grid)
        points_small
    ))
    
    # CRUCIAL: Preparar os pontos com Latitude e Longitude fixas
    all_sample_points = all_sample_points.map(prepare_points_for_export)

    # 5. Extração com neighborhoodToArray
    # Como os pontos já existem, isso é rápido.
    kernel = ee.Kernel.rectangle(PATCH_SIZE//2, PATCH_SIZE//2, 'pixels')
    patches_array = ee.Image(stack_image).addBands(alertas_img)#.neighborhoodToArray(kernel)

    # Definição dos Seletores (O que vai entrar no arquivo)
    # Incluímos latitude e longitude explicitamente
    selectors = t0_bands + t1_bands + ['label', 'latitude', 'longitude']

    size_Alerts = all_sample_points.size().getInfo()
    print("número de aletas a sere empacotados ", size_Alerts)
    if size_Alerts > MAX_PATCHES_PER_SHARD + 5:
        # Estimativa de shards.
        # Assumindo 5000 pontos max por grid. 5000/300 = ~17 shards.
        # Se você quiser ser preciso, precisaria do size(), mas isso demora.
        
        NUM_SHARDS = max(1, int(size_Alerts / MAX_PATCHES_PER_SHARD) + 1)
        print(f"🚀 Alertas totais: {size_Alerts}. Exportando em {NUM_SHARDS} shards de segurança.")

        # Gerar Shards usando randomColumn para garantir distribuição
        all_sample_points = all_sample_points.randomColumn('split')
        
        for i in range(int(NUM_SHARDS)):
            min_val = i / NUM_SHARDS
            max_val = (i + 1) / NUM_SHARDS
            
            # Filtra a fatia
            shard_points = all_sample_points.filter(
                                        ee.Filter.And(
                                            ee.Filter.gte('split', min_val),
                                            ee.Filter.lt('split', max_val)
                                        ))

            fname = f"tfrecord_{wrs_name}_{nyear}_{digit_month}_{nregion}_part_{i:03d}"
            try:
                # O segredo do predict: geometries=False (usamos as colunas lat/lon)
                # tileScale=16 para aliviar a RAM do worker do GEE
                shard_samples = patches_array.neighborhoodToArray(kernel).sampleRegions(
                    collection= shard_points,
                    scale= SCALE,
                    geometries= False, 
                    tileScale= 16      # CRUCIAL para evitar estouro de memória com arrays
                )                
                
                task = ee.batch.Export.table.toDrive(
                    collection=shard_samples,
                    description=fname,
                    folder=EXPORT_FOLDER,
                    fileFormat='TFRecord',
                    selectors=selectors
                )
                
                task.start()
                print(f"     -> Task enviada: {fname}")

            except Exception as e:
                print(f"     ❌ Erro ao configurar shard {i}: {e}")

    else:
        print(f" número de alertas {size_Alerts}  Iniciando exportação ...") 
        
        try:
            shard_samples = patches_array.sampleRegions(
                collection= all_sample_points,
                scale= SCALE,
                geometries= False, # TFRecord não precisa de lat/lon, só dos pixels
                tileScale= 16      # CRUCIAL para evitar estouro de memória com arrays
            )
            
            fname = f"tfrecord_{wrs_name}_{nyear}_{digit_month}_{nregion}_part_0"   
            task = ee.batch.Export.table.toDrive(
                    collection=shard_samples,
                    description=fname,
                    folder=EXPORT_FOLDER,
                    fileFormat='TFRecord',
                    selectors=selectors
                )
                
            task.start()

            print(f"     -> Task enviada: {fname}")
        except:
                print(" >>> erro de no task >>>> ")    

        # sys.exit()

# ==============================================================================
# 4. INTEGRAÇÃO NO SEU LOOP EXISTENTE
# ==============================================================================


# 1. Listar todos os assets no folder
asset_list = ee.data.listAssets(ASSET_ALERTS_FOLDER)  # , maxResults= 1000
asset_list = ee.Dictionary(asset_list).get('assets').getInfo()
print("tamanho N ", len(asset_list))

# Carrega o mapeamento
with open('mapping_ID_to_Region.json', 'r') as f:
    grid_map = json.load(f)

# 0: 15  30: 50 50: 60  60: 72Dez conta 07
# 15: 30  Dez conta 04
pos_inic = 20
pos_end = 30
for cc, nasset in enumerate(asset_list[pos_inic: pos_end]):
    print(f" # {cc + pos_inic}/72 >> {nasset['id'].split("/")[-1]} ")
    
    partes = nasset['id'].split("/")[-1].split("_")
    wrs_path = partes[2]
    wrs_row = partes[3]

    current_id = f"{wrs_path}/{wrs_row}"
    # Pega o nome da região para compor o nome do arquivo de saída
    region_name = grid_map.get(current_id, "regiao_desconhecida")

    featAlerts = ee.FeatureCollection(nasset['id'])
    numbAlerts = featAlerts.size().getInfo()
    print(f"Size of alerts {numbAlerts} in feature collection path/Row >>> {current_id}")   
    
    
    # print("geometria quadrada com area ", float(ee.Geometry(geomGroupAlerts).area(0.1).getInfo()) / 10000)
    # sys.exit()
    predict_stack = get_prediction_stack(wrs_path, wrs_row, DATA_INIC)
    # sys.exit()
    # 2. Iniciar exportação de imagens (TFRecords) em vez de tabela
    export_patches_for_predict(featAlerts, predict_stack, f"{wrs_path}_{wrs_row}", region_name)
    print(f"✅ Tarefa de exportação da ftrecord iniciada com sufixo {wrs_path}_{wrs_row}")
    
    # sys.exit()