// =====================================================================
// Visualização de Biomas x Regiões de Amostragem Fotovoltaica - MapBiomas
// =====================================================================
// Sobrepõe os limites dos biomas (MapBiomas Col.11) com o shapefile de
// regiões de amostragem (buffer 5 km) usado no pipeline fotovoltaico,
// para conferência visual e print de tela antes da exportação dos patches.

var ASSET_BIOMAS  = 'projects/mapbiomas-workspace/AUXILIAR/bioma_2025_e250k_5kbuffer';
var ASSET_REGIONS = 'projects/mapbiomas-arida/energias/shp_revisao2_16_05_2026_buffer_fotovoltaic_5km';

var feat_biomas = ee.FeatureCollection(ASSET_BIOMAS);
var SHP_REGIONS = ee.FeatureCollection(ASSET_REGIONS);

print('Biomas - quantidade:', feat_biomas.size());
print('Biomas - 1ª feature:', feat_biomas.first());
print('Regiões - quantidade:', SHP_REGIONS.size());
print('Regiões - 1ª feature:', SHP_REGIONS.first());

// ---- Camada categórica de biomas -------------------------------------

var LISTA_BIOMAS  = ['Amazônia', 'Caatinga', 'Cerrado', 'Mata Atlântica', 'Pampa', 'Pantanal'];
var PALETA_BIOMAS = ['2ca25f', 'd95f0e', 'fec44f', '31a354', '9ecae1', '3182bd'];

var biomas_cod = feat_biomas.map(function(feat) {
  var idx = ee.List(LISTA_BIOMAS).indexOf(feat.get('NAME')).add(1); // 0 = não encontrado
  return feat.set('id_bioma', idx);
});

var img_biomas = biomas_cod
  .filter(ee.Filter.gt('id_bioma', 0))
  .reduceToImage(['id_bioma'], ee.Reducer.first())
  .rename('bioma');

var visBiomas = {
  min: 1, max: LISTA_BIOMAS.length,
  palette: PALETA_BIOMAS
};

// ---- Camada de regiões (contorno) ------------------------------------

var visRegioes = { color: 'ffff00', fillColor: '00000000', width: 2 };

// ---- Adicionar ao mapa -------------------------------------------------

Map.setOptions('SATELLITE');

Map.addLayer(img_biomas.selfMask(), visBiomas, 'Biomas (MapBiomas Col.11)');
Map.addLayer(SHP_REGIONS.style(visRegioes), {}, 'Regiões de amostragem (buffer 5 km)');

Map.centerObject(SHP_REGIONS, 5);

// ---- Legenda ------------------------------------------------------------

var legend = ui.Panel({
  style: { position: 'bottom-left', padding: '8px 12px' }
});

legend.add(ui.Label({ value: 'Biomas x Regiões de amostragem FV',
  style: { fontWeight: 'bold', fontSize: '13px', margin: '0 0 6px 0' } }));

var addLegendRow = function(color, label) {
  var box = ui.Label({
    style: { backgroundColor: color, padding: '6px', margin: '2px 6px 2px 0' }
  });
  var desc = ui.Label({ value: label, style: { margin: '2px 0' } });
  legend.add(ui.Panel([box, desc], ui.Panel.Layout.Flow('horizontal')));
};

addLegendRow('2ca25f', 'Amazônia');
addLegendRow('d95f0e', 'Caatinga');
addLegendRow('fec44f', 'Cerrado');
addLegendRow('31a354', 'Mata Atlântica');
addLegendRow('9ecae1', 'Pampa');
addLegendRow('3182bd', 'Pantanal');
addLegendRow('ffff00', 'Regiões de amostragem (buffer 5 km)');

Map.add(legend);

// =====================================================================
// (Opcional) Exportar um "print" da composição (biomas + contorno das
// regiões) como imagem estática para o Drive, em vez de usar o ícone de
// câmera do Code Editor. Ajuste EXPORT_REGION antes de rodar.
// =====================================================================

var TIRAR_PRINT = false;

if (TIRAR_PRINT) {
  var EXPORT_REGION = SHP_REGIONS.geometry().bounds();
  var EXPORT_SCALE  = 500; // m/pixel — ajustar conforme a escala do print desejado

  var composicao = ee.ImageCollection([
    img_biomas.selfMask().visualize(visBiomas),
    SHP_REGIONS.style(visRegioes).visualize()
  ]).mosaic();

  Export.image.toDrive({
    image: composicao,
    description: 'print_biomas_regioes_fv',
    folder: 'MapBiomas_exports',
    region: EXPORT_REGION,
    scale: EXPORT_SCALE,
    maxPixels: 1e10
  });

  print('Task de export "print_biomas_regioes_fv" criada — rode em Tasks.');
}
