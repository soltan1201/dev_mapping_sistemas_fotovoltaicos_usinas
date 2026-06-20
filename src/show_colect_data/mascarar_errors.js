// ============================================================
// visualizar_fotovoltaicas.js  — v2
// Comparador lado a lado de backbones — Planet NICFI + predições
// ============================================================

// ============================================================
// 1. ASSETS
// ============================================================
var asset_output_shp = 'projects/mapbiomas-arida/energias';
function processoExportar(ROIsFeat, nameB){
    var assetIdss =  asset_output_shp + '/' + nameB;
    var optExp = {
          'collection': ROIsFeat, 
          'description': nameB, 
          'assetId': assetIdss          
        };
    Export.table.toAsset(optExp) ;
    print("salvando ... " + nameB + "..!")  ;  
}


// var asset_regioesBuffer = 'projects/mapbiomas-arida/update_02_05_2026_buffer_fotovoltaic_5km';

var asset_regioesBuffer = 'projects/mapbiomas-arida/energias/shp_revisao2_16_05_2026_buffer_fotovoltaic_5km';
var asset_NICFI         = 'projects/planet-nicfi/assets/basemaps/americas';
var asset_predicoes     = 'projects/geo-data-s/assets/fotovoltaica/usinas_br_gc';
var asset_col10 = "projects/mapbiomas-brazil/assets/LAND-COVER/COLLECTION-10/SOLAR-PANELS/classification";
//# aqui estão os poligons complementares
var ASSET_LIMIT_ROTULOS_2024 = "projects/mapbiomas-arida/energias/polygons_base_paneis_fotovoltaicos_16_05_2026";
var ASSET_EXCLUSION_2024     = "projects/mapbiomas-arida/energias/poligons_exclusion_comision_16_05_2026";
var modelo = 'unet';
var backbone = 'efficientnetb7';
var nyear = 2024;
var col_usinaFV = ee.FeatureCollection(asset_regioesBuffer);
print("know metadados ", col_usinaFV.limit(5));
print("Shoiw quantidade" , col_usinaFV.size());
// ============================================================
// 2. REGIÕES DISPONÍVEIS
// ============================================================
var REGIOES = [
    "00000000000000000000","00000000000000000001","00000000000000000002","00000000000000000003",
    "00000000000000000004","00000000000000000005","00000000000000000006","00000000000000000007",
    "00000000000000000008","00000000000000000009","0000000000000000000a","0000000000000000000b",
    "0000000000000000000c","0000000000000000000d","0000000000000000000e","0000000000000000000f",
    "00000000000000000010","00000000000000000011","00000000000000000012","00000000000000000013",
    "00000000000000000014","00000000000000000015","00000000000000000016","00000000000000000017",
    "00000000000000000018","00000000000000000019","0000000000000000001a","0000000000000000001b",
    "0000000000000000001c","0000000000000000001d","0000000000000000001e","0000000000000000001f",
    "00000000000000000020","00000000000000000021","00000000000000000022","00000000000000000023",
    "00000000000000000024","00000000000000000025","00000000000000000026","00000000000000000027",
    "00000000000000000028","00000000000000000029","0000000000000000002a","0000000000000000002b",
    "0000000000000000002c","0000000000000000002d","0000000000000000002e","0000000000000000002f",
    "00000000000000000030","00000000000000000031","00000000000000000032","00000000000000000033",
    "00000000000000000034","00000000000000000035","00000000000000000036","00000000000000000037",
    "00000000000000000038","00000000000000000039","0000000000000000003a","0000000000000000003b",
    "0000000000000000003c","0000000000000000003d","0000000000000000003e","0000000000000000003f",
    "00000000000000000040","00000000000000000041","00000000000000000042","00000000000000000043",
    "00000000000000000044","00000000000000000045","00000000000000000046","00000000000000000047",
    "00000000000000000048","00000000000000000049","0000000000000000004a","0000000000000000004b",
    "0000000000000000004c","0000000000000000004d","0000000000000000004e","0000000000000000004f",
    "00000000000000000050","00000000000000000051","00000000000000000052","00000000000000000053",
    "00000000000000000054","00000000000000000055","00000000000000000056","00000000000000000057",
    "00000000000000000058","00000000000000000059"
];

// ============================================================
// 4. PARÂMETROS DE VISUALIZAÇÃO
// ============================================================
var visRGB    = {bands: ['R', 'G', 'B'], min: 100,   max: 3200,  gamma: 1.8};

var visPredE  = {min: 0, max: 1, palette: ['#FF0000']};
var visPredD  = {min: 0, max: 1, palette: ['#0066FF']};
var visBuffer = {color: '#FFD700', fillColor: '00000000', width: 1};


// ============================================================
// 5. MAPAS ESQUERDO E DIREITO
// ============================================================


var rotulos_limit = ee.FeatureCollection(ASSET_LIMIT_ROTULOS_2024);
var areas_exclusao = ee.FeatureCollection(ASSET_EXCLUSION_2024);
print("show metadados poliongs de exclussão ", areas_exclusao);
var visualizar= true;

// classes_2025FV convertida para imagem — usada para preencher buracos da segmentação em 2025
var img_classes_2025FV = ee.Image(0).paint(classes_2025FV, 1).selfMask().rename('b1');

[2024, 2025].forEach(function(year) {
    // Mosaico Planet NICFI — 2º semestre (jul–dez)
    var mosaic    = ee.ImageCollection(asset_NICFI)
                        .filter(ee.Filter.date(year + '-07-01', year + '-12-31'))
                        .min()
                        .clip(col_usinaFV.geometry());

    var predFV = ee.ImageCollection(asset_predicoes)
                    .filter(ee.Filter.eq('year', year))
                    .filter(ee.Filter.eq('modelo', modelo))
                    .filter(ee.Filter.eq('backbone', backbone))
                    .mosaic().selfMask();

    // Para 2025: mescla classes desenhadas (buracos da segmentação) ao predFV
    if (year === 2025) {
        predFV = ee.ImageCollection([predFV, img_classes_2025FV]).mosaic();
    }

    print("Coletion Backbone " + backbone + " " + year, predFV);
    Map.addLayer(mosaic,          visRGB,    '01. Planet RGB ' + String(year),       visualizar);
    if (year === nyear){ 
        var pred_col10 = ee.ImageCollection(asset_col10)
                      .filter(ee.Filter.eq('year', nyear))
                      .first();       

    Map.addLayer(pred_col10, visPredE, "pred Col10 " + nyear,         visualizar);
    }
    Map.addLayer(predFV,     visPredD, '09. ' + modelo + ' / ' + backbone + ' ' + year, visualizar);
    visualizar = false;
});

Map.addLayer(rotulos_limit,   {color: 'red'},   'rotulos base');
Map.addLayer(areas_exclusao,  {color: 'black'}, 'area_exclusao');
Map.addLayer(col_usinaFV,     visBuffer, '10. Buffer 5 km',      true);

// ============================================================
// 6. CAMADAS DE GEOMETRIAS DESENHADAS — exportação e visualização
// ============================================================

// --- incluir_2025_c10 (11 pol) ---
// bounds de cada polígono + merge com as 90 regiões do buffer (que já traz as properties)
var incluir_2025_c10_bounds = incluir_2025_c10.map(function(f) {
  return ee.Feature(f.geometry().bounds());
});
var regioes_atualizadas = ee.FeatureCollection(asset_regioesBuffer).merge(incluir_2025_c10_bounds);
processoExportar(regioes_atualizadas, 'shp_revisao3_10_06_2026_buffer_fotovoltaic_5km');
// Map.addLayer(incluir_2025_c10,        {color: '00FFFF'}, 'incluir_2025_c10 (11 pol)',   false);

// --- classes_2025FV (162 pol class 1) — preenchimento de buracos da segmentação ---
processoExportar(classes_2025FV,      'polygons_base_paneis3_FV_10_06_2026');
// Map.addLayer(classes_2025FV,          {color: '00FF00'}, 'classes_2025FV (162 pol)',    false);

// --- merge_col11_col10 (26 ptos) — ptos onde as regiões da col10 será unida à col11 ---
processoExportar(merge_col11_col10,   'region_with_merge_layer_col11_col10');
// Map.addLayer(merge_col11_col10,       {color: 'FFA500'}, 'merge_col11_col10 (26 ptos)', false);

// --- excluir_analises (7 ptos) — sem segmentação, revisados, sem FV ---
processoExportar(excluir_analises,    'regions_to_excluir_analises_10_06_2026');
// Map.addLayer(excluir_analises,        {color: 'FF4444'}, 'excluir_analises (7 ptos)',   false);

// --- repetir24 (2 ptos) — só 2024 deve incluir (col10 melhor que col11) ---
processoExportar(repetir24,           'region_with_layer_colection10_24_10_06_2026');
// Map.addLayer(repetir24,               {color: 'FFFF00'}, 'repetir24 (2 ptos)',           false);

// --- ampliar_exc (319 pol FV class 1) — exclui pixels col10/col11 em todos os anos ---
processoExportar(ampliar_exc,         'poligons_exclusion_comision_v2_10_06_2026');
// Map.addLayer(ampliar_exc,             {color: 'AA00FF'}, 'ampliar_exc (319 pol)',        false);

// --- rem_exc (14 ptos) — remove polígonos da camada de exclusão já salva ---
// Referência: projects/mapbiomas-arida/energias/poligons_exclusion_comision_16_05_2026
areas_exclusao = areas_exclusao.filter(ee.Filter.bounds(rem_exc).not());
processoExportar(areas_exclusao,             'poligons_exclusion_comision_v1_10_06_2026');
// Map.addLayer(rem_exc,                 {color: 'FFFFFF'}, 'rem_exc (14 ptos)',            false);

