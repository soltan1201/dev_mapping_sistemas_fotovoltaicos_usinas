var asset_usinaFV = "users/mapbiomascaatinga04/atualiz_buffer_fotovoltaic_5km";
var asset_photovoltaica = 'projects/mapbiomas-arida/fotovoltaic_rural';
var asset_fv_b5k = "projects/mapbiomas-arida/fotovoltaic_rural_buffer_5KM"
var colection_fv = ee.FeatureCollection(asset_photovoltaica);
var coleFV5km = ee.FeatureCollection(asset_fv_b5k);
coleFV5km = coleFV5km.union(0.01);

print(" colection_fv ", colection_fv.size());
print("show metadata ", colection_fv.limit(5));

var col_usinaFV = ee.FeatureCollection(asset_usinaFV);
print(" colection Usinas Fotovoltaicas ", col_usinaFV.size());
print("show metadata ", col_usinaFV.limit(5));

Map.addLayer(colection_fv, {color: 'red'}, 'Fotovoltaica');
Map.addLayer(coleFV5km, {color: 'black'}, 'Fotovoltaica');
Map.addLayer(col_usinaFV, {color: 'blue'}, 'Usina FV');

colection_fv = colection_fv.map(function(feat){return feat.buffer(10000).bounds()});


Export.table.toAsset({
    collection: coleFV5km, 
    description: "fotovoltaic_rural_buffer_5KMmerg", 
    assetId: 'projects/mapbiomas-arida/fotovoltaic_rural_buffer_5KMmerg'
})

var asset_br = 'users/CartasSol/shapes/Brasil_Manual';
var shp_BR = ee.FeatureCollection(asset_br);

coleFV5km = coleFV5km.map(function(feat){return feat.set('id_cod', 1)})
var mask_coleFV5km = coleFV5km.reduceToImage(['id_cod'], ee.Reducer.first());

Map.addLayer(mask_coleFV5km.selfMask(), {max:1}, 'mask');
var name_exp = 'fotovoltaic_rural_buffer_5KM_mask'
var optExp = {   
    'image': mask_coleFV5km.byte().selfMask(), 
    'description': name_exp, 
    'assetId': 'projects/mapbiomas-arida/' + name_exp, 
    'region': shp_BR.getInfo()['coordinates'], 
    'scale': 30, 
    'maxPixels': 1e13,
    "pyramidingPolicy": {".default": "mode"}
}
Export.image.toAsset(optExp)