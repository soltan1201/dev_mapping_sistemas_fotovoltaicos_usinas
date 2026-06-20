// ============================================================
// toolkit_col11_visualizador.js
// Visualiza 10 anos (2016–2025) de predições FV por região
// + gráfico de série temporal de área (ha)
// ============================================================
// USO:
//   1. Cole no GEE Code Editor
//   2. Selecione uma região no painel esquerdo
//   3. Os 10 mapas são preenchidos com NICFI + FV do respectivo ano
//   4. O gráfico calcula e exibe a área FV (ha) por ano
// ============================================================

// ============================================================
// CONFIGURAÇÕES
// ============================================================

var ASSET_LAYERS  = 'projects/mapbiomas-workspace/AMOSTRAS/col11/CAATINGA/layers_energia';
var ASSET_REGIONS = 'projects/mapbiomas-arida/energias/shp_revisao3_10_06_2026_buffer_fotovoltaic_5km';
var ASSET_NICFI   = 'projects/planet-nicfi/assets/basemaps/americas';
var ASSET_COL10   = 'projects/mapbiomas-brazil/assets/LAND-COVER/COLLECTION-10/SOLAR-PANELS/classification';

var VERSION   = 1;
var YEARS     = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025];
var VIS_FV    = {min: 0, max: 1, palette: ['#FF4400']};
var VIS_COL10 = {min: 0, max: 1, palette: ['#0055FF']};  // azul — diferencia Col10 do FV col11
var VIS_RGB   = {bands: ['R', 'G', 'B'], min: 100, max: 3200, gamma: 1.8};
var SCALE     = 4.77;  // m/pixel — Planet NICFI

// ============================================================
// 10 MAPAS (um por ano) + LINKER
// ============================================================

var mapWidgets = [];
for (var k = 0; k < YEARS.length + 1; k++) {  // +1 para Col10 2024
  var m = ui.Map();
  m.setControlVisibility({all: false, zoomControl: false});
  m.style().set({stretch: 'both'});
  mapWidgets.push(m);
}

// Sincroniza zoom/pan entre todos os mapas
var mapLinker = ui.Map.Linker(mapWidgets);

// ============================================================
// GRADE DE MAPAS — 4 linhas × 3 colunas
// ============================================================

var mapGrid = ui.Panel([], ui.Panel.Layout.flow('vertical'), {stretch: 'both'});

for (var row = 0; row < 4; row++) {
  var rowWidgets = [];
  for (var col = 0; col < 3; col++) {
    var idx = row * 3 + col;
    if (idx < mapWidgets.length) rowWidgets.push(mapWidgets[idx]);
  }
  var rowPanel = ui.Panel(
    rowWidgets,
    ui.Panel.Layout.flow('horizontal'),
    {stretch: 'horizontal', height: '240px'}
  );
  mapGrid.add(rowPanel);
}

// ============================================================
// PAINEL DE CONTROLE — esquerda
// ============================================================

var statusLabel  = ui.Label('Carregando regiões...', {color: '#666', fontSize: '11px', margin: '4px 0'});
var regionSelect = ui.Select({placeholder: '--- selecione uma região ---', onChange: onRegionSelected});
var chartPanel   = ui.Panel();

var controlPanel = ui.Panel(
  [
    ui.Label('FV Col11 — Visualizador Regional', {
      fontWeight: 'bold', fontSize: '14px', margin: '0 0 10px 0'
    }),
    ui.Label('Região (id_region):', {fontWeight: 'bold', margin: '0 0 2px 0'}),
    regionSelect,
    statusLabel,
    ui.Label('Série temporal — Área FV (ha):', {
      fontWeight: 'bold', margin: '16px 0 4px 0'
    }),
    chartPanel
  ],
  ui.Panel.Layout.flow('vertical'),
  {width: '310px', padding: '10px'}
);

// ============================================================
// LAYOUT RAIZ
// ============================================================

ui.root.clear();
ui.root.setLayout(ui.Panel.Layout.flow('horizontal'));
ui.root.add(controlPanel);
ui.root.add(mapGrid);

// ============================================================
// CARREGA LISTA DE REGIÕES
// ============================================================

var regionsFC = ee.FeatureCollection(ASSET_REGIONS);
var lst_idsregionsFC = regionsFC.reduceColumns(
  ee.Reducer.toList(),
  ['system:index']).get('list').getInfo();
lst_idsregionsFC.sort();
regionSelect.items().reset(lst_idsregionsFC);
statusLabel.setValue(lst_idsregionsFC.length + ' regiões disponíveis. Selecione uma.');

// ============================================================
// CALLBACK — seleção de região
// ============================================================

function onRegionSelected(regionId) {
  statusLabel.setValue('Carregando ' + regionId + '...');
  chartPanel.clear();

  var geom = regionsFC
    .filter(ee.Filter.eq('system:index', regionId))
    .first()
    .geometry();

  YEARS.forEach(function(year, i) {
    var assetId = ASSET_LAYERS + '/solar-panel-' + year + '-' + regionId + '-' + VERSION;

    var nicfi = ee.ImageCollection(ASSET_NICFI)
      .filter(ee.Filter.date(year + '-07-01', (year + 1) + '-01-01'))
      .min()
      .clip(geom);

    var fv = ee.Image(assetId).select('b1').selfMask();

    // clear() remove camadas e widgets; re-adiciona o label do ano
    mapWidgets[i].clear();
    mapWidgets[i].add(ui.Label(String(year), {
      fontSize: '12px',
      fontWeight: 'bold',
      backgroundColor: 'rgba(0,0,0,0.60)',
      color: '#FFFFFF',
      padding: '2px 7px',
      position: 'top-left'
    }));

    mapWidgets[i].centerObject(geom, 13);
    mapWidgets[i].addLayer(nicfi, VIS_RGB, 'NICFI ' + year, true);
    mapWidgets[i].addLayer(fv,    VIS_FV,  'FV '    + year, true);
  });

  // Mapa Col10 2024 — índice 10, azul para diferenciar do FV col11
  var nicfi2024 = ee.ImageCollection(ASSET_NICFI)
    .filter(ee.Filter.date('2024-07-01', '2025-01-01'))
    .min()
    .clip(geom);
  var col10img = ee.ImageCollection(ASSET_COL10)
    .filter(ee.Filter.eq('year', 2024))
    .first()
    .selfMask()
    .reproject({crs: 'EPSG:4326', scale: SCALE});

  mapWidgets[10].clear();
  mapWidgets[10].add(ui.Label('Col10 2024', {
    fontSize: '12px', fontWeight: 'bold',
    backgroundColor: 'rgba(0,40,140,0.75)', color: '#FFFFFF',
    padding: '2px 7px', position: 'top-left'
  }));
  mapWidgets[10].centerObject(geom, 13);
  mapWidgets[10].addLayer(nicfi2024, VIS_RGB,   'NICFI 2024', true);
  mapWidgets[10].addLayer(col10img,  VIS_COL10, 'Col10 2024', true);

  computeAreaChart(regionId, geom);
}

// ============================================================
// SÉRIE TEMPORAL DE ÁREA (ha)
// ============================================================

function computeAreaChart(regionId, geom) {
  statusLabel.setValue('Calculando áreas...');

  var areaCollection = ee.ImageCollection(
    YEARS.map(function(year) {
      var assetId = ASSET_LAYERS + '/solar-panel-' + year + '-' + regionId + '-' + VERSION;
      var fv      = ee.Image(assetId).select('b1').selfMask();
      return ee.Image.pixelArea()
               .divide(10000)          // m² → ha
               .updateMask(fv)
               .rename('area_ha')
               .set('year', year)
               .set('system:time_start', ee.Date.fromYMD(year, 1, 1).millis());
    })
  );

  var chart = ui.Chart.image.series({
    imageCollection: areaCollection,
    region:          geom,
    reducer:         ee.Reducer.sum(),
    scale:           SCALE,
    xProperty:       'year'
  })
  .setChartType('LineChart')
  .setOptions({
    title:     'Área FV (ha) — ' + regionId,
    hAxis:     {title: 'Ano', format: '####', gridlines: {count: 10}},
    vAxis:     {title: 'Área (ha)', minValue: 0},
    colors:    ['#FF4400'],
    pointSize: 5,
    lineWidth: 2,
    legend:    {position: 'none'},
    chartArea: {width: '78%', left: '16%', top: '12%', bottom: '14%'}
  });

  chartPanel.add(chart);
  statusLabel.setValue('Pronto — ' + regionId);
}
