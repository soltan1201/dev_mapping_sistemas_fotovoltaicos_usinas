
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Produzido por Geodatin - Dados e Geoinformacao
DISTRIBUIDO COM GPLv2
@author: geodatin
"""
import os
import subprocess
import ee
import time
import sys
import json
import collections
from pathlib import Path
collections.Callable = collections.abc.Callable

try:
    ee.Initialize(project= 'geo-data-s') # project='ee-cartassol'  #  'geo-data-s' # 'mapbiomas-caatinga-cloud02'
    print('The Earth Engine package initialized successfully!')
except ee.EEException as e:
    print('The Earth Engine package failed to initialize!')
except:
    print("Unexpected error:", sys.exc_info()[0])
    raise

#exporta a imagem classificada para o asset
def ExportarMapsLabelAsset(mapaRF,  nomeDesc, mgeom):    
    pathIC = "projects/geo-data-s/assets/fotovoltaica/version_3"
    idasset =  pathIC + "/" + nomeDesc
    optExp = {
        'image': mapaRF, 
        'description': nomeDesc, 
        'assetId':idasset, 
        'region': mgeom.getInfo()['coordinates'],
        'scale': 4, 
        'crs': "EPSG:3857",
        'maxPixels': 1e9,
        "pyramidingPolicy":{".default": "mode"}
    }
    task = ee.batch.Export.image.toAsset(**optExp)
    task.start() 
    print("salvando ... " + nomeDesc + "..!")



param = {
    'asset_fotovol_predv2' : 'projects/geo-data-s/assets/fotovoltaica/version_2',
    'asset_fotovol_predv1': 'projects/geo-data-s/assets/fotovoltaica/versao1',
    'asset_fotovol_output' : 'projects/geo-data-s/assets/fotovoltaica/version_3',
    'assetNICFI': 'projects/planet-nicfi/assets/basemaps/americas',
    'asset_brasil_manual': 'users/CartasSol/shapes/brasil_manual',
    "asset_biomas_raster" : 'projects/mapbiomas-workspace/AUXILIAR/biomas-raster-41',
    "asset_polygons_FotoV": 'projects/mapbiomas-workspace/AMOSTRAS/col9/CAATINGA/Energias/polygons/polygons_base_paneis_fotovoltaicos_19_11',
    "asset_regions_FotoV": 'projects/mapbiomas-workspace/AMOSTRAS/col9/CAATINGA/Energias/polygons/regions_on_paneis_fotovoltaicos_19_11',
    "asset_regions_errorV1": 'projects/mapbiomas-workspace/AMOSTRAS/col9/CAATINGA/Energias/polygons/regions_erro_predict_fotovoltaicos_08_12',
    "asset_regions_errorV2": 'projects/mapbiomas-workspace/AMOSTRAS/col9/CAATINGA/Energias/polygons/regions_erro_predict_fotovoltaicos_18_12',
    'asset_output_Polygons': 'projects/mapbiomas-workspace/AMOSTRAS/col9/CAATINGA/Energias/polygons',    
}
version = '2';

bioCaat = ee.Image(param['asset_biomas_raster']).gt(0);
fotovol_predv2 = (
    ee.ImageCollection(param['asset_fotovol_predv2'])
         .map(lambda img: img.toByte())
)

fotovol_predv1 = (
    ee.ImageCollection(param['asset_fotovol_predv1'])
         .map(lambda img: img.toByte())
)
fotovol_predv1 = fotovol_predv1.max().unmask(0)

shp_errorsV1 = ee.FeatureCollection(param['asset_regions_errorV1']);
print("shp of errors V1", shp_errorsV1.size().getInfo());                   
shp_errorsV2 = ee.FeatureCollection(param['asset_regions_errorV2']);
print("shp of errors V2", shp_errorsV2.size().getInfo());  

shp_errorsV2 = shp_errorsV2.merge(shp_errorsV1).map(lambda feat: feat.set('classe', 1));
print("shp of errors V2 merge ", shp_errorsV2.size().getInfo()); 
shp_errors_img = shp_errorsV2.reduceToImage(['classe'], ee.Reducer.first())
shp_errors_img = shp_errors_img.unmask(0)
lstCod = fotovol_predv2.reduceColumns(ee.Reducer.toList(), ['system:index']).get('list').getInfo()


for ncod in lstCod[:]:
    im_tmp = ee.Image(
                fotovol_predv2.filter(ee.Filter.eq('system:index', ncod))
                    .first())
    geom = im_tmp.geometry();
    imC1 = fotovol_predv1.clip(geom);
    errors = shp_errors_img.clip(geom)
    imC2 = im_tmp.add(imC1);
    imC2 = imC2.subtract(errors);
    imC2 = imC2.gt(0).set(
                    'data_rev', '16-02-2025',
                    'remotion_errors', True,
                    'version', version,
                    'tipo_image', True
                );
    name_export = ncod + '_v' + version;
    ExportarMapsLabelAsset(imC2, name_export, geom)


# fotovol_pred = cleaning_layer(fotovol_predv1, fotovol_predv2, shp_errorsV2)
