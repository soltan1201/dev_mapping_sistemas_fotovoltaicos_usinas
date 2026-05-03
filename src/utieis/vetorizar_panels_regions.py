import ee
import os
import sys
import collections
collections.Callable = collections.abc.Callable
from pathlib import Path


try:
    ee.Initialize(project= 'geo-data-s')
    print('The Earth Engine package initialized successfully!')
except ee.EEException as e:
    print('The Earth Engine package failed to initialize!')
except:
    print("Unexpected error:", sys.exc_info()[0])
    raise


def exportarVectorClassAlerta(FeatSHP, nameAls):
    
    optExp = {
        'collection': FeatSHP, 
        'description': nameAls, 
        'assetId':  params['pathOut'] + '/' + nameAls        
    }

    task = ee.batch.Export.table.toAsset(**optExp)
    
    task.start()
    print (task.status() )
    print ("salvando ... !", nameAls)


# https://code.earthengine.google.com/1fb397b216989800bbc89c0b0d7b1635
def convertImgToVector(imgFV, geom):    
    optVect = {            
            'geometry': geom, 
            'scale': 3, 
            'geometryType': 'polygon',
            'bestEffort': True, 
            'maxPixels': 1e13, 
            # 'tileScale': 4, 
            'geometryInNativeProjection': True
        }
    # filter Reject manchas com grupos desconectados
    maskRuidoReject = ee.Image(imgFV).toInt32().connectedComponents(ee.Kernel.square(2), 3).select('labels')      
    imgAlrcDef = imgFV.where(maskRuidoReject.neq(0), 0)

    # retornando um feat
    # featTemp = imgAlrcDef.updateMask(imgAlrc).reduceToVectorsStreaming(**optVect)
    featTemp = imgAlrcDef.selfMask().reduceToVectorsStreaming(**optVect)
    return featTemp
    
    return ee.FeatureCollection([])


params = {
    'asset_Col9' : "projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_integration_v1",
    'asset_fotovol_predv2' : 'projects/geo-data-s/assets/fotovoltaica/version_3',  
    'pathOut': 'projects/geo-data-s/assets/fotovoltaica/shp/vetor_reg',
    #'asset_colection_sentinel': 'projects/mapbiomas-public/assets/brazil/lulc/collection_S2_beta/collection_LULC_S2_beta'
    'asset_colection_sentinel': 'projects/mapbiomas-public/assets/brazil/lulc_10m/collection2/mapbiomas_10m_collection2_integration_v1'
}


fotovol_predv2 = ee.ImageCollection(params['asset_fotovol_predv2']);
lstCod = fotovol_predv2.reduceColumns(ee.Reducer.toList(), ['system:index']).get('list').getInfo()

for ncod in lstCod[:]:
    raster_tmp = fotovol_predv2.filter(ee.Filter.eq('system:index', ncod)).first()
    print("show metadata ", raster_tmp.get('system:index').getInfo())
    reg_raster = raster_tmp.geometry().bounds()

    vectorUn = convertImgToVector(raster_tmp, reg_raster) 
    vectorUn = vectorUn.map(lambda feat: feat.set(
                                        'startDate' , '01-10-2024',
                                        'sensor', 'Planet',
                                        'mes', 10,
                                        'ano', 2024,                                                                              
                                        # 'area', feat.area(),
                                        'tipo', 'alerta',
                                        'version', '2.0'
                            ))

    print("vetorizando region  >> ", ncod)                        
                
    exportarVectorClassAlerta(vectorUn, ncod)