
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
def processoExportar(mapaRF,  nomeDesc, pathIC):    
    # pathIC = "projects/geo-data-s/assets/fotovoltaica/version_2"
    idasset =  os.path.join(pathIC, nomeDesc)
    optExp = {
        'image': mapaRF, 
        'description': nomeDesc, 
        'assetId':idasset, 
        # 'region': mapaRF.geometry().getInfo()['coordinates'],
        'scale': 4, 
        'crs': "EPSG:3857",
        'maxPixels': 1e9,
        "pyramidingPolicy":{".default": "mode"}
    }
    task = ee.batch.Export.image.toAsset(**optExp)
    task.start() 
    print("salvando ... " + nomeDesc + "..!")

dict_models_ver = {
    "tif_fotovoltaicav1": "unet_resnet50",
    "tif_fotovoltaicav2": "unet_resnet101",
    "tif_fotovoltaicav3": "unet_resnet152",
    "tif_fotovoltaicav4": "unet_mobilenet",
    "tif_fotovoltaicav5": "unet_resnext50",
    "tif_fotovoltaicav6": "unet_xception",
}


createFolder = False
createIC = False
version = 1
backbone = 'resnet50'
model = 'unet'
pathsIC = "pprojects/geo-data-s/assets/fotovoltaica/usinas_br"
# ******************** Command Line Instructions ******************************#
# Here we create a folder in GEE using earthengine tool
# references https://cloud.google.com/sdk/gcloud/reference/storage
if createFolder:
    folderDir = 'projects/geo-data-s/assets/fotovoltaica'
    comando = "earthengine create folder " + folderDir
    os.system(comando)
    print(f" folder in asset < {comando} > create !" )
# Create a collection (it looks like another folder, but internally it will be
# an image collection).
if createIC:    
    comando = "earthengine create collection " + pathsIC
    os.system(comando)
    print(f" image Collection < {comando} > create !" )


# ******************** Bash script ********************************************#

gcBucket= "mapbiomas-energia"
folderDirGS = 'fotovoltaicas_tif/tif_fotovoltaicav1'   # entrada como argumento

model = dict_models_ver[folderDirGS.split("/")[-1].split("_")[0]]
backbone = dict_models_ver[folderDirGS.split("/")[-1].split("_")[0]]

comando = f"gcloud storage ls gs://{gcBucket}/{folderDirGS}*"
# lstdirsFileGS = os.system(comando)  # os.system
# print(lstdirsFileGS)
processo = subprocess.check_output(comando, shell=True)
# tmp = processo.read()
# print(processo)
# print(type(processo))
lstdirsFileGS = str(processo.decode('utf-8')).split("\n")
data_inic = ee.Date.fromYMD(2016,12, 31)
dictProp = {        
    'year': 2016,
    'version': str(version), 
    'semestre':  2, 
    'backbone': backbone,
    'modelo': model,
    'system:time_start': data_inic
}

# Get file names to extract date and call ingestion command for each file to be added into an asset as image collection
for cc, pathdir in enumerate(lstdirsFileGS[:]):
    name_tif = pathdir.split("/")[-1]
    nyear = name_tif.split("_")[-1][:4]
    data_inic = ee.Date.fromYMD(nyear,12, 31)
    namefile = f"{name_tif.replace('pred', 'reg')}_{model}_{backbone}"

    print(f" #{cc} >> {pathdir}")            
    idAsset = os.path.join(pathsIC, namefile)
    print("idAsset >> ", idAsset)
    
    # https://developers.google.com/earth-engine/guides/command_line
    # earthengine upload image --asset_id=users/myuser/asset --pyramiding_policy=sample --nodata_value=255 gs://bucket/image.tif
    newComand = f"earthengine upload image --asset_id={idAsset} --time_start={data_inic} --pyramiding_policy=sample  --nodata_value=-9999 {pathdir}"
    #  --property={dictProp}
    # os.system(newComand)
    print(" processing ... ")
    imgTIF = ee.Image.loadGeoTIFF(pathdir)
    imgTIF = ee.Image(imgTIF).set(
            'year', nyear, 
            'version', str(version), 
            'backbone', backbone,
            'modelo', model,
            'semestre',  2,         
            'system:time_start', data_inic)
    if cc > -1:
        processoExportar(imgTIF, namefile.replace('.tif', ''), pathsIC)
