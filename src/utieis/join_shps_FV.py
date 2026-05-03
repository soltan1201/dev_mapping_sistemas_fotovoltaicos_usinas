import ee
import sys
from tqdm import tqdm
import collections
collections.Callable = collections.abc.Callable

try:
    ee.Initialize(project= 'geo-data-s')
    print('The Earth Engine package initialized successfully!')
except ee.EEException as e:
    print('The Earth Engine package failed to initialize!')
except:
    print("Unexpected error:", sys.exc_info()[0])
    raise

param = {
    'asset_SHPs' : {'id': 'projects/geo-data-s/assets/fotovoltaica/shp/vetor_reg'},
    'output_shpJoin': 'projects/geo-data-s/assets/fotovoltaica/shp'
}

# salva ftcol para um assetindexIni
def save_ROIs_toAsset(collection, name):
    outputAsset = param['output_shpJoin'] + "/" + name
    optExp = {
        'collection': collection,
        'description': name,
        'assetId': outputAsset 
    }

    # task = ee.batch.Export.table.toAsset(**optExp)
    # task.start()

    optExp = {
        'collection': collection,
        'description': name,
        'folder': 'shps_FV',
        'fileFormat':  "SHP"
    }

    task = ee.batch.Export.table.toDrive(**optExp)
    task.start()
    print(f"⚡️⚡ exportando ROIs da bacia << {name} >> ...! ⚡️⚡")

def GetPolygonsfromFolder(dictAsset):   

    getlistPtos = ee.data.getList(dictAsset)
    ColectionPtos = ee.FeatureCollection([])
    
    for idAsset in tqdm(getlistPtos):         
        print("join asset ", idAsset.get('id').replace(dictAsset['id'], "..."))
        feattmp = ee.FeatureCollection(idAsset.get('id'))    
        ColectionPtos = ColectionPtos.merge(feattmp)
        
    return ee.FeatureCollection(ColectionPtos)


featRegions = GetPolygonsfromFolder(param['asset_SHPs'])

namexp = 'shp_regions_fotoVoltaica_2024_v2'
save_ROIs_toAsset(ee.FeatureCollection(featRegions), namexp)