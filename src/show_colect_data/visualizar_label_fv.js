// =====================================================================
// Visualização do Label Fotovoltaico - MapBiomas
// =====================================================================

var year = 2024;
var ASSET_AREA_SAMPLES = "projects/mapbiomas-arida/energias/shp_area_fotovoltaic_samples_update_16_05_2026";
var ASSET_EXCLUSION_2024  = "projects/mapbiomas-arida/energias/poligons_exclusion_comision_16_05_2026";
var ASSET_LIMIT_ROTULOS_2024 = "projects/mapbiomas-arida/energias/polygons_base_paneis_fotovoltaicos_16_05_2026";
var ASSET_LABEL_2024      = 'projects/geo-data-s/assets/fotovoltaica/usinas_br_gc';
var ASSET_POINTS_2024 = 'projects/mapbiomas-arida/energias/pontos_areas_DB_16_05_2026';


// ---- Camadas --------------------------------------------------------

var base_FV_complementar = ee.Image(0)
  .paint(ee.FeatureCollection(ASSET_LIMIT_ROTULOS_2024), 1)
  .byte();

var mask_negativa = ee.Image(0)
  .paint(ee.FeatureCollection(ASSET_EXCLUSION_2024), 1)
  .byte();

var mylabel = ee.ImageCollection(ASSET_LABEL_2024)
                      .filter(ee.Filter.eq('year', year))
                      .filter(ee.Filter.eq('modelo', 'unet'))
                      .filter(ee.Filter.eq('backbone', 'resnet50'))
                      .filter(ee.Filter.neq('formato', 'tfr'))
                      .mosaic()
                      .add(base_FV_complementar)
                      .gte(1)
                      .updateMask(mask_negativa.eq(0))
                      .unmask(0)
                      .rename('label')
                      .toByte();

// ---- Paletas de visualização ----------------------------------------

var visLabel = {
  min: 0, max: 1,
  palette: ['000000', 'ffff00']   // preto = 0, amarelo = 1 (painel FV)
};

var visExclusao = { color: 'ff0000', fillColor: 'ff000033' }; // vermelho
var visRotulos  = { color: '0000ff', fillColor: '0000ff22' }; // azul

// ---- Adicionar ao mapa ----------------------------------------------

Map.setOptions('SATELLITE');
// Map.setCenter(-47.5, -15.8, 6);

Map.addLayer(
  ee.FeatureCollection(ASSET_EXCLUSION_2024),
  visExclusao,
  'Máscara de exclusão'
);

Map.addLayer(
    ee.FeatureCollection(ASSET_LIMIT_ROTULOS_2024),
    visRotulos,
    'Polígonos base (rótulos)'
);

Map.addLayer(
  mylabel.selfMask(),
  visLabel,
  'Label FV ' + year
);


Map.addLayer(
    ee.FeatureCollection(ASSET_AREA_SAMPLES)
          .filterBounds(ee.FeatureCollection(ASSET_POINTS_2024).geometry()),
    {color: 'yellow'},
    'areas de coleta'
)
// ---- Legenda --------------------------------------------------------

var legend = ui.Panel({
  style: { position: 'bottom-left', padding: '8px 12px' }
});

legend.add(ui.Label({ value: 'Label Fotovoltaico ' + year,
  style: { fontWeight: 'bold', fontSize: '13px', margin: '0 0 6px 0' } }));

var addLegendRow = function(color, label) {
  var box = ui.Label({
    style: { backgroundColor: color, padding: '6px', margin: '2px 6px 2px 0' }
  });
  var desc = ui.Label({ value: label, style: { margin: '2px 0' } });
  legend.add(ui.Panel([box, desc], ui.Panel.Layout.Flow('horizontal')));
};

addLegendRow('ffff00', 'Painel fotovoltaico (label = 1)');
addLegendRow('000000', 'Não fotovoltaico (label = 0)');
addLegendRow('ff0000', 'Área de exclusão');
addLegendRow('0000ff', 'Polígonos base (rótulos)');

Map.add(legend);

print('Coleção de labels disponíveis:', ee.ImageCollection(ASSET_LABEL_2024).limit(5));
print('Label ' + year + ' - info:', mylabel);
