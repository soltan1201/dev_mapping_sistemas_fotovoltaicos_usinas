var vis = {
    mosaico: {
        bands: ["R","G","B"],
        min: 145, max: 1700
    }
}
// 1. Definições Iniciais
var asset_regs = 'users/CartasSol/shapes/Brasil_Manual'
var regions = ee.FeatureCollection(asset_regs);
print("shoiw metadata of regions ", regions);
var ASSET_NICFI   = 'projects/planet-nicfi/assets/basemaps/americas'
var year = 2024;
// Mapeamento padrão L8/L9 no asset de 32 dias
var bandNames = ['B', 'G', 'R', 'N'];

// 1. Função de Scaling Robusta (Linear Stretch entre percentis)
// Esta função garante que o range [p5, p90] ocupe todo o espectro [0, 1]
var applyRobustScaling = function(image, geometry) {
    
    // Calculamos os percentis para TODAS as bandas de uma vez
    var stats = image.reduceRegion({
        reducer: ee.Reducer.percentile([1, 99]), // Aumentei para 95 para evitar que fique muito clara/estourada
        geometry: geometry,
        scale: 150,
        maxPixels: 1e13,
        bestEffort: true
    });
    print(stats)
    var scaledBands = bandNames.map(function(bandName) {
        var bStr = ee.String(bandName);
        var pLow = ee.Number(stats.get(bStr.cat('_p1')));
        var pHigh = ee.Number(stats.get(bStr.cat('_p99')));
        
        var band = image.select(bStr);
        // Clamp para garantir que valores fora do percentil não criem artefatos < 0 ou > 1
        return {'pLow': pLow, 'pHigh': pHigh, 'band_name': bStr};
    });
    
    return ee.List(scaledBands);
};
// 3. Processamento por Ano e Período

var yearCurrent = ee.Number(2024);

var dictPer = {
     'start': ee.Date.fromYMD(yearCurrent, 6, 1), 
     'end': ee.Date.fromYMD(yearCurrent, 12, 31), 
     'suffix': 'year' 
};

var collection = ee.ImageCollection(ASSET_NICFI)
                    .filterDate(dictPer.start, dictPer.end)
                    .median().clip(regions.geometry()); // Redutor para remover nuvens residuais e outliers
print("metadados coleção ", collection);
// Aplicar scaling baseado na geometria das regiões (pode ser pesado, usando clip/bounds)
var dict_percentieis = applyRobustScaling(collection, regions.geometry().bounds());
print(dict_percentieis)

Map.addLayer(collection, vis.mosaico, 'planet')