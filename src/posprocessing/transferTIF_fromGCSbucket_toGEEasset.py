
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



createFolder = False
createIC = False
version = 4
mes = 10
# pathIC = "projects/geo-data-s/assets/fotovoltaica/version_3"
pathsIC = f"projects/geo-data-s/assets/fotovoltaica/version_{version}"
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
folderDirGS = 'fotovoltaicas_tif/'

comando = f"gcloud storage ls gs://{gcBucket}/{folderDirGS}*"
# lstdirsFileGS = os.system(comando)  # os.system
# print(lstdirsFileGS)
processo = subprocess.check_output(comando, shell=True)
# tmp = processo.read()
# print(processo)
# print(type(processo))
lstdirsFileGS = str(processo.decode('utf-8')).split("\n")
for nyear in range(2015, 2026):
    for mes in [6, 7, 9, 10, 11]:
        data_inic = ee.Date.fromYMD(nyear,mes, 1)
        dictProp = {        
            'year': nyear,
            'month': mes,
            'version': str(version), 
            'data_inic':  data_inic, 
            'data_end':  data_inic.advance(1,'month'), 
            'system:time_start': data_inic
        }
        vdate = data_inic
        # Get file names to extract date and call ingestion command for each file to be added into an asset as image collection
        for cc, pathdir in enumerate(lstdirsFileGS[:]):
            namefile = pathdir.split("/")[-1]
            if "_g2d.tif" in pathdir and f"_{mes}_" in pathdir and str(nyear) in pathdir:  # 
                print(f" #{cc} >> {pathdir}")            
                idAsset = os.path.join(pathsIC, namefile)
                print("idAsset >> ", idAsset)
                
                # https://developers.google.com/earth-engine/guides/command_line
                # earthengine upload image --asset_id=users/myuser/asset --pyramiding_policy=sample --nodata_value=255 gs://bucket/image.tif
                newComand = f"earthengine upload image --asset_id={idAsset} --time_start={vdate} --pyramiding_policy=sample  --nodata_value=-9999 {pathdir}"
                #  --property={dictProp}
                # os.system(newComand)
                print(" processing ... ")
                imgTIF = ee.Image.loadGeoTIFF(pathdir)
                imgTIF = ee.Image(imgTIF).set(
                        'year', nyear, 'month', mes,
                        'version', str(version), 'data_inic',  data_inic, 
                        'data_end',  data_inic.advance(1,'month'),         
                        'system:time_start', data_inic)
                if cc > -1:
                    processoExportar(imgTIF, namefile.replace('.tif', ''), pathsIC)
