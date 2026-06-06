#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Produzido por Geodatin - Dados e Geoinformacao
DISTRIBUIDO COM GPLv2
@author: geodatin
"""
import os
import ee
from tqdm import tqdm
import sys
import copy
import collections
import pandas as pd
pd.set_option("mode.copy_on_write", True)
from pathlib import Path
collections.Callable = collections.abc.Callable
pathparent = str(Path(os.getcwd()).parents[0])
sys.path.append(pathparent)
from configure_account_projects_ee import get_current_account, get_project_from_account
from gee_tools import *
projAccount = get_current_account()
print(f"projetos selecionado >>> {projAccount} <<<")

try:
    ee.Initialize(project= projAccount) # project='ee-cartassol'  #  'geo-data-s' # 'mapbiomas-caatinga-cloud02'
    print('The Earth Engine package initialized successfully!')
except ee.EEException as e:
    print('The Earth Engine package failed to initialize!')
except:
    print("Unexpected error:", sys.exc_info()[0])
    raise

def set_properties(img):
    sysInd = img.id()
    parteId = ee.String(sysInd).split("_")
    idCod = ee.List(parteId).get(0)
    nyear = ee.List(parteId).get(1)
    nyear = ee.Algorithms.If(
                    ee.Algorithms.IsEqual(ee.String(nyear), ee.String('pred')), 
                    ee.Number(2024), 
                    ee.Number.parse(nyear)
                ) 
    return img.set(
        'idCod', idCod,
        'year', nyear,
        'version', 2,
        'data_inic', ee.Date.fromYMD(nyear, 10, 1),
        'data_end', ee.Date.fromYMD(nyear, 12, 31)

    ).toByte() 

def calcular_area(img, layerArea, geom):
    layerArea = layerArea.updateMask(img.eq(1).rename('classe')).clip(geom)
    optRed = {
        'reducer': ee.Reducer.sum(),
        'geometry': geom,
        'scale': 4,
        'bestEffort': True, 
        'maxPixels': 1e13
    }    
    areas = layerArea.reduceRegion(**optRed)
    # print(areas.getInfo())
    return areas.get('area')#.getInfo()

#exporta a imagem classificada para o asset
def processoExportar(layerFV, geomet, nameB):
    idasset =  param['asset_output'] + "/" + nameB
    optExp = {
        'image': layerFV, 
        'description': nameB, 
        'assetId':idasset, 
        'region':geomet.getInfo()['coordinates'], #
        'scale': 4, 
        'maxPixels': 1e13,
        "pyramidingPolicy":{".default": "mode"},
        # 'priority': 1000
    }
    task = ee.batch.Export.image.toAsset(**optExp)
    task.start() 
    print("salvando ... " + nameB + "..!")
    # print(task.status())
    for keys, vals in dict(task.status()).items():
        print ( "  {} : {}".format(keys, vals))

param = {
    'asset_input': 'projects/geo-data-s/assets/fotovoltaica/version_2',
    'asset_inputC': 'projects/geo-data-s/assets/fotovoltaica/version_3',
    'asset_output': 'projects/geo-data-s/assets/fotovoltaica/version_2_clean',
    'lstYear': [2015,2016,2017,2018,2019,2020,2021,2022,2023,2024] ,
    'lstYearInv': [2024,2023,2022,2021,2020,2019,2018,2017,2016,2015]  
}

fotovol_predv2 = ee.ImageCollection(param['asset_input'])
fotovol_clean = ee.ImageCollection(param['asset_inputC'])
print("imagens Collection size cleaning ", fotovol_clean.size().getInfo())
# print(fotovol_clean.first().getInfo())

lstRegions = fotovol_predv2.reduceColumns(ee.Reducer.toList(), ['system:index']).get('list').getInfo()
lstIdCod = []
for id_reg in tqdm(lstRegions):
    # print("região ", id_reg)
    idCod = id_reg.split("_")[0]
    if idCod not in lstIdCod:
        lstIdCod.append(idCod)
print(f"we load {len(lstIdCod)} ids regions ")
fotovol_pred = fotovol_predv2.map(lambda img: set_properties(img))
fotovolClean = fotovol_clean.map(lambda img: set_properties(img))
# 00000000000000000027
# 00000000000000000041
activo = False
for ii, idCod in enumerate(lstIdCod[2:]):    
    print(f"#{ii}  >> processing idCod: {idCod}")
    layersCod = fotovol_pred.filter(ee.Filter.eq('idCod', idCod))
    layersCodCl = fotovolClean.filter(ee.Filter.eq('idCod', idCod))
    # print("número de Images loaded ", layersCod.size().getInfo())
    # print("quantidade selecionada ",layersCodCl.size().getInfo())
    geomC = layersCod.first().geometry()
    pixelArea = ee.Image.pixelArea().divide(10000).clip(geomC)
    anterior = ee.Image().byte()
    # sys.exit()
    if activo:
        for cc, nyear in enumerate(param['lstYearInv']):
            if cc == 0:
                layersCodYY = layersCodCl.first()   
                layersCodYY = (layersCodYY.focalMax(radius= 1, kernelType= 'square')
                                        .focalMin(radius= 1, kernelType= 'square'))         
            else:
                layersCodYY = layersCod.filter(ee.Filter.eq('year', nyear)).first()
                layersCodYY = (layersCodYY.focalMax(radius= 1, kernelType= 'square')
                                        .focalMin(radius= 1, kernelType= 'square'))            
                layersCodYY = layersCodYY.multiply(anterior)
            
            
            arealayer = calcular_area(layersCodYY, pixelArea, geomC)            
            print(f"#{nyear} Area do layer = ")   # {arealayer}
            anterior = copy.deepcopy(layersCodYY)        
            layersCodYY = layersCodYY.set(
                                'area', arealayer,
                                'idCod', idCod,
                                'year', nyear,
                                'month', 10
                                )
            name_layer = f"{idCod}_{nyear}_pred"
            processoExportar(layersCodYY, geomC.bounds(), name_layer)

    if idCod == '00000000000000000041':
        activo = True



