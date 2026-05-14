// ============================================================
// visualizar_fotovoltaicas.js  — v2
// Comparador lado a lado de backbones — Planet NICFI + predições
// ============================================================

// ============================================================
// 1. ASSETS
// ============================================================
var asset_regioesBuffer = 'projects/mapbiomas-arida/update_02_05_2026_buffer_fotovoltaic_5km';
var asset_NICFI         = 'projects/planet-nicfi/assets/basemaps/americas';
var asset_predicoes     = 'projects/geo-data-s/assets/fotovoltaica/usinas_br_gc';

var col_usinaFV = ee.FeatureCollection(asset_regioesBuffer);

// ============================================================
// 2. REGIÕES DISPONÍVEIS
// ============================================================
var REGIOES = [
  '00000000000000000021','00000000000000000022','00000000000000000023',
  '00000000000000000024','00000000000000000025','00000000000000000026',
  '00000000000000000027','00000000000000000028','00000000000000000029',
  '0000000000000000002a','0000000000000000002b','0000000000000000002c',
  '0000000000000000002d','0000000000000000002e','0000000000000000002f',
  '00000000000000000030','00000000000000000031','00000000000000000032',
  '00000000000000000033','00000000000000000034','00000000000000000035',
  '00000000000000000036','00000000000000000037','00000000000000000038',
  '00000000000000000039','0000000000000000003a','0000000000000000003b',
  '0000000000000000003c','0000000000000000003d','0000000000000000003e',
  '0000000000000000003f','00000000000000000040','00000000000000000041',
  '00000000000000000042','00000000000000000043','00000000000000000044',
  '00000000000000000045','00000000000000000046','00000000000000000047',
  '00000000000000000048','00000000000000000049','0000000000000000004a',
  '0000000000000000004b','0000000000000000004c','0000000000000000004d',
  '0000000000000000004e','0000000000000000004f','00000000000000000050',
  '00000000000000000051','00000000000000000052','00000000000000000053',
  '00000000000000000054','00000000000000000055','00000000000000000056',
  '00000000000000000057'
];

// ============================================================
// 3. ÍNDICES ESPECTRAIS
// ============================================================
var calcularIndices = function(img) {
  var b    = img.select(['B', 'G', 'R', 'N']);
  var pvi  = img.expression('float((R - N) / (R + N))', {
    R: b.select('R'), N: b.select('N')
  }).add(1).multiply(10000).rename('pvi');
  var ndvi = img.normalizedDifference(['N', 'R']).multiply(10000).rename('ndvi');
  return img.select(['B', 'G', 'R', 'N']).toInt16()
            .addBands([pvi.toInt16(), ndvi.toInt16()]);
};

// ============================================================
// 4. PARÂMETROS DE VISUALIZAÇÃO
// ============================================================
var visRGB    = {bands: ['R', 'G', 'B'], min: 100,   max: 3200,  gamma: 1.8};
var visFalsa  = {bands: ['N', 'R', 'G'], min: 100,   max: 4000,  gamma: 1.6};
var visPVI    = {min: 4000,  max: 14000, palette: ['#1a9641', '#ffffbf', '#d7191c']};
var visNDVI   = {min: -3000, max: 8000,  palette: ['#8B4513', '#ffffff', '#00aa00']};
var visPredE  = {min: 0, max: 1, palette: ['#FF0000']};
var visPredD  = {min: 0, max: 1, palette: ['#0066FF']};
var visBuffer = {color: '#FFD700', fillColor: '00000000', width: 1};

// ============================================================
// 5. MAPAS ESQUERDO E DIREITO
// ============================================================
var mapLeft  = ui.Map();
var mapRight = ui.Map();
mapLeft.setOptions('SATELLITE');
mapRight.setOptions('SATELLITE');
mapLeft.setControlVisibility({layerList: true, zoomControl: true, mapTypeControl: false});
mapRight.setControlVisibility({layerList: true, zoomControl: false, mapTypeControl: false});

// ============================================================
// 6. FUNÇÃO DE RENDERIZAÇÃO
// ============================================================
var renderizar = function(year, regiaoId, modeloE, backboneE, modeloD, backboneD) {
  mapLeft.layers().reset();
  mapRight.layers().reset();

  var roi = regiaoId === 'Todas'
    ? col_usinaFV.geometry()
    : col_usinaFV.filter(ee.Filter.eq('id', regiaoId)).geometry();

  // Mosaico Planet NICFI — 2º semestre (jul–dez), anual
  var mosaic    = ee.ImageCollection(asset_NICFI)
    .filter(ee.Filter.date(year + '-07-01', year + '-12-31'))
    .median()
    .clip(roi);
  var mosaicIdx = calcularIndices(mosaic);

  // Helper: filtra predições por modelo, backbone e (opcional) região
  var getPred = function(modelo, backbone) {
    var col = ee.ImageCollection(asset_predicoes)
      .filter(ee.Filter.eq('year', year))
      .filter(ee.Filter.eq('modelo', modelo));
    if (backbone !== 'nenhum') {
      col = col.filter(ee.Filter.eq('backbone', backbone));
    }
    if (regiaoId !== 'Todas') {
      col = col.filter(ee.Filter.eq('region', regiaoId));
    }
    return col.mosaic().selfMask();
  };

  var predE = getPred(modeloE, backboneE);
  var predD = getPred(modeloD, backboneD);

  // --- Mapa esquerdo ---
  mapLeft.addLayer(mosaic,                    visRGB,   '01. Planet RGB',       true);
  mapLeft.addLayer(mosaic,                    visFalsa, '02. Falsa Cor',        false);
  mapLeft.addLayer(mosaicIdx.select('ndvi'),  visNDVI,  '03. NDVI',             false);
  mapLeft.addLayer(mosaicIdx.select('pvi'),   visPVI,   '04. PVI',              false);
  mapLeft.addLayer(predE,                     visPredE, '05. ' + modeloE + ' / ' + backboneE, true);
  mapLeft.addLayer(col_usinaFV,               visBuffer,'06. Buffer 5 km',      true);

  // --- Mapa direito ---
  mapRight.addLayer(mosaic,                   visRGB,   '01. Planet RGB',       true);
  mapRight.addLayer(mosaic,                   visFalsa, '02. Falsa Cor',        false);
  mapRight.addLayer(mosaicIdx.select('ndvi'), visNDVI,  '03. NDVI',             false);
  mapRight.addLayer(mosaicIdx.select('pvi'),  visPVI,   '04. PVI',              false);
  mapRight.addLayer(predD,                    visPredD, '05. ' + modeloD + ' / ' + backboneD, true);
  mapRight.addLayer(col_usinaFV,              visBuffer,'06. Buffer 5 km',      true);

  mapLeft.centerObject(roi, regiaoId === 'Todas' ? 6 : 10);
};

// ============================================================
// 7. PAINEL DE CONTROLE (overlay no mapa esquerdo)
// ============================================================

var titulo    = ui.Label('Comparador FV', {
  fontWeight: 'bold', fontSize: '15px', margin: '0 0 2px 0'
});
var subtitulo = ui.Label('Usinas Fotovoltaicas — MapBiomas', {
  fontSize: '10px', color: '#555', margin: '0 0 10px 0'
});

// --- Ano ---
var labelAno  = ui.Label('Ano:', {fontWeight: 'bold', margin: '0 0 3px 0'});
var selectAno = ui.Select({
  items: ['2016','2017','2018','2019','2020','2021','2022','2023','2024'],
  value: '2024',
  style: {stretch: 'horizontal'}
});

// --- Região ---
var labelReg  = ui.Label('Região:', {fontWeight: 'bold', margin: '6px 0 3px 0'});
var selectReg = ui.Select({
  items: ['Todas'].concat(REGIOES),
  value: 'Todas',
  style: {stretch: 'horizontal'}
});

// --- Separador ---
var mkSep = function() {
  return ui.Label('━━━━━━━━━━━━━━━━━━━', {
    color: '#bbb', fontSize: '9px', margin: '8px 0 6px 0'
  });
};

// --- Mapa esquerdo ---
var labelE  = ui.Label('◀  ESQUERDO', {
  fontWeight: 'bold', fontSize: '11px', color: '#c0392b', margin: '0 0 4px 0'
});
var labelME = ui.Label('Modelo:', {fontSize: '11px', margin: '0 0 2px 0'});
var selME   = ui.Select({
  items: ['unet', 'deeplabv3plus', 'pspnet'],
  value: 'unet',
  style: {stretch: 'horizontal'}
});
var labelBE = ui.Label('Backbone:', {fontSize: '11px', margin: '4px 0 2px 0'});
var selBE   = ui.Select({
  items: ['resnet50','resnet101','resnet152','resnext50','mobilenet','xception','nenhum'],
  value: 'resnet50',
  style: {stretch: 'horizontal'}
});

// --- Mapa direito ---
var labelD  = ui.Label('▶  DIREITO', {
  fontWeight: 'bold', fontSize: '11px', color: '#1a6eb5', margin: '0 0 4px 0'
});
var labelMD = ui.Label('Modelo:', {fontSize: '11px', margin: '0 0 2px 0'});
var selMD   = ui.Select({
  items: ['unet', 'deeplabv3plus', 'pspnet'],
  value: 'unet',
  style: {stretch: 'horizontal'}
});
var labelBD = ui.Label('Backbone:', {fontSize: '11px', margin: '4px 0 2px 0'});
var selBD   = ui.Select({
  items: ['resnet50','resnet101','resnet152','resnext50','mobilenet','xception','nenhum'],
  value: 'resnet101',
  style: {stretch: 'horizontal'}
});

// --- Botão + Status ---
var statusLabel = ui.Label('Selecione e clique em Aplicar.', {
  fontSize: '10px', color: '#666', margin: '6px 0 0 0', whiteSpace: 'wrap'
});

var btnAplicar = ui.Button({
  label: 'Aplicar',
  style: {stretch: 'horizontal', margin: '8px 0 0 0', backgroundColor: '#2b6cb0', color: '#fff'},
  onClick: function() {
    statusLabel.setValue('Carregando...');
    var year = parseInt(selectAno.getValue(), 10);
    var reg  = selectReg.getValue();
    var mE   = selME.getValue();
    var bE   = selBE.getValue();
    var mD   = selMD.getValue();
    var bD   = selBD.getValue();
    renderizar(year, reg, mE, bE, mD, bD);
    statusLabel.setValue(
      year + ' | ' + (reg === 'Todas' ? 'Todas regiões' : reg) +
      '\n◀ ' + mE + '/' + bE + '  ▶ ' + mD + '/' + bD
    );
  }
});

// --- Legenda ---
var mkLegRow = function(cor, texto) {
  var row = ui.Panel({layout: ui.Panel.Layout.flow('horizontal'), style: {margin: '2px 0'}});
  row.add(ui.Label({style: {backgroundColor: cor, padding: '6px', margin: '0 5px 0 0', border: '1px solid #ccc'}}));
  row.add(ui.Label(texto, {fontSize: '10px', margin: '1px 0'}));
  return row;
};
var legTitle = ui.Label('Legenda', {fontWeight: 'bold', fontSize: '11px', margin: '10px 0 4px 0'});

// --- Painel montado ---
var painel = ui.Panel({
  widgets: [
    titulo, subtitulo,
    labelAno,  selectAno,
    labelReg,  selectReg,
    mkSep(),
    labelE, labelME, selME, labelBE, selBE,
    mkSep(),
    labelD, labelMD, selMD, labelBD, selBD,
    mkSep(),
    btnAplicar,
    statusLabel,
    legTitle,
    mkLegRow('#FF0000', 'Predição — esquerdo'),
    mkLegRow('#0066FF', 'Predição — direito'),
    mkLegRow('#FFD700', 'Buffer 5 km'),
  ],
  style: {
    position: 'top-left',
    width: '230px',
    padding: '10px',
    backgroundColor: '#f9f9f9'
  }
});

// ============================================================
// 8. MONTAGEM FINAL
// ============================================================
ui.root.clear();
ui.root.add(mapLeft);
ui.root.add(mapRight);

ui.Map.Linker([mapLeft, mapRight], 'change-bounds');

mapLeft.add(painel);

// ============================================================
// 9. RENDERIZAÇÃO INICIAL
// ============================================================
renderizar(2024, 'Todas', 'unet', 'resnet50', 'unet', 'resnet101');
statusLabel.setValue('2024 | Todas regiões\n◀ unet/resnet50  ▶ unet/resnet101');
