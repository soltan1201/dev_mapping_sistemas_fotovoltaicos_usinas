// ============================================================
// visualizar_fotovoltaicas.js
// Visualização de usinas fotovoltaicas — Planet NICFI + predições
// Toolkit interativo: seletor de ano, semestre e região
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

var asset_regioesBuffer = 'projects/mapbiomas-arida/update_02_05_2026_buffer_fotovoltaic_5km';
var asset_NICFI         = 'projects/planet-nicfi/assets/basemaps/americas';
var asset_predicoes     = 'projects/geo-data-s/assets/fotovoltaica/usinas_br_gc';

var col_usinaFV = ee.FeatureCollection(asset_regioesBuffer);
print("Area de mapeamento Brasil ", col_usinaFV);
var asset_polygons_FotoV = 'projects/mapbiomas-workspace/AMOSTRAS/col9/CAATINGA/Energias/polygons/polygons_base_paneis_fotovoltaicos_19_11';
var shp_polFV_in = ee.FeatureCollection(asset_polygons_FotoV);
print("shp de poligonso de rotulos ", shp_polFV_in);



// ============================================================
// 2. ÍNDICES ESPECTRAIS
// ============================================================
var calcularIndices = function(img) {
    var b = img.select(['B', 'G', 'R', 'N']);

    var pvi = img.expression('float((R - N) / (R + N))', {
        R: b.select('R'), N: b.select('N')
    }).add(1).multiply(10000).rename('pvi');

    var ndvi = img.normalizedDifference(['N', 'R'])
        .multiply(10000).rename('ndvi');

    return img.select(['B', 'G', 'R', 'N']).toInt16()
        .addBands([pvi.toInt16(), ndvi.toInt16()]);
};

// ============================================================
// 3. PARÂMETROS DE VISUALIZAÇÃO
// ============================================================
var visRGB     = {bands: ['R', 'G', 'B'], min: 100,   max: 3200,  gamma: 1.8};
var visFalsa   = {bands: ['N', 'R', 'G'], min: 100,   max: 4000,  gamma: 1.6};
var visPVI     = {min: 4000,  max: 14000, palette: ['#1a9641', '#ffffbf', '#d7191c']};
var visNDVI    = {min: -3000, max: 8000,  palette: ['#8B4513', '#ffffff', '#00aa00']};
var visPred    = {min: 0, max: 1, palette: ['#FF0000']};
var visBuffer  = {color: '#FFD700', fillColor: '00000000', width: 1};

// ============================================================
// 4. FUNÇÃO PRINCIPAL DE RENDERIZAÇÃO
// ============================================================
var renderizar = function(year, semestre, regiaoId) {

    Map.layers().reset();

    // Região de interesse
    var roi = regiaoId === 'Todas'
        ? col_usinaFV.geometry()
        : col_usinaFV.filter(ee.Filter.eq('id', regiaoId)).geometry();

    // Datas Planet NICFI
    var dateStart = semestre === '1º Semestre'
        ? (year + '-01-01') : (year + '-07-01');
    var dateEnd   = semestre === '1º Semestre'
        ? (year + '-06-30') : (year + '-12-31');

    // Mosaico Planet
    var mosaic = ee.ImageCollection(asset_NICFI)
        .filter(ee.Filter.date(dateStart, dateEnd))
        .median()
        .clip(roi);

    var mosaicIdx = calcularIndices(mosaic);
    var modelo_segm = 'unet';
    var modelo_backbone = 'resnet50';  // mobilenet: resnet101:  resnet152: resnet50: resnext50
    // Predições
    var pred_col = ee.ImageCollection(asset_predicoes)
                      .filter(ee.Filter.eq('year', year))
                      .filter(ee.Filter.eq('modelo', modelo_segm))
                      .filter(ee.Filter.eq('backbone', modelo_backbone))
                      .filter(ee.Filter.neq('formato', 'tfr'));
    print(pred_col)
    print(pred_col.aggregate_histogram('backbone'))
    var pred =  pred_col.mosaic().gte(0.5).selfMask();

    // Adicionar camadas
    Map.addLayer(mosaic,                     visRGB,    '01. Planet RGB',         true);
    Map.addLayer(mosaic,                     visFalsa,  '02. Falsa Cor',          false);
    Map.addLayer(mosaicIdx.select('ndvi'),   visNDVI,   '03. NDVI',               false);
    Map.addLayer(mosaicIdx.select('pvi'),    visPVI,    '04. PVI (Painel Solar)', false);
    Map.addLayer(pred,                       visPred,   '05. Predição > 0.5',     true);
    Map.addLayer(col_usinaFV,                visBuffer, '06. Buffer 5 km',        true);
    Map.addLayer(shp_polFV_in, {color: '554369'}, 'poligonos', false);
    // Map.centerObject(roi, regiaoId === 'Todas' ? 6 : 10);
};

// ============================================================
// 5. TOOLKIT — painel lateral
// ============================================================

// --- Título ---
var titulo = ui.Label('Visualizador FV', {
    fontWeight: 'bold',
    fontSize: '16px',
    margin: '0 0 10px 0',
    color: '#1a1a1a'
});

var subtitulo = ui.Label('Usinas Fotovoltaicas — MapBiomas', {
    fontSize: '11px',
    color: '#555',
    margin: '0 0 14px 0'
});

// --- Seletor de ANO ---
var labelAno = ui.Label('Ano:', {fontWeight: 'bold', margin: '0 0 4px 0'});
var selectAno = ui.Select({
    items: ['2016','2017','2018','2019','2020','2021','2022','2023','2024'],
    value: '2024',
    style: {stretch: 'horizontal'}
});

// --- Seletor de SEMESTRE ---
var labelSem = ui.Label('Semestre (Planet NICFI):', {fontWeight: 'bold', margin: '8px 0 4px 0'});
var selectSem = ui.Select({
    items: ['1º Semestre', '2º Semestre'],
    value: '2º Semestre',
    style: {stretch: 'horizontal'}
});

// --- Seletor de REGIÃO ---
var labelReg = ui.Label('Região:', {fontWeight: 'bold', margin: '8px 0 4px 0'});

// Lista de regiões a partir das features do asset (carregado assincronamente)
var selectReg = ui.Select({
    items: ['Todas'],
    value: 'Todas',
    style: {stretch: 'horizontal'}
});

// Popula o seletor de regiões com IDs únicos do asset
col_usinaFV.aggregate_array('id').evaluate(function(ids) {
    if (!ids) {
        selectReg.items().reset(['Todas']);
        return;
    }
    var uniq = ['Todas'].concat(ids.sort().map(String));
    selectReg.items().reset(uniq);
});

// --- Botão APLICAR ---
var btnAplicar = ui.Button({
    label: 'Aplicar',
    style: {
        stretch: 'horizontal',
        margin: '14px 0 0 0',
        backgroundColor: '#2b6cb0',
        color: '#fff'
    },
    onClick: function() {
        statusLabel.setValue('Carregando...');
        var year = parseInt(selectAno.getValue(), 10);
        var sem  = selectSem.getValue();
        var reg  = selectReg.getValue();
        renderizar(year, sem, reg);
        statusLabel.setValue(
            'Exibindo: ' + year + ' | ' + sem + ' | Região: ' + reg
        );
    }
});

// --- Status ---
var statusLabel = ui.Label('Selecione os parâmetros e clique em Aplicar.', {
    fontSize: '11px',
    color: '#666',
    margin: '8px 0 0 0',
    whiteSpace: 'wrap'
});

// --- Legenda de cores ---
var legendaTitulo = ui.Label('Legenda', {
    fontWeight: 'bold',
    fontSize: '12px',
    margin: '16px 0 6px 0'
});

var mkLegRow = function(cor, texto) {
    var row = ui.Panel({layout: ui.Panel.Layout.flow('horizontal'), style: {margin: '2px 0'}});
    row.add(ui.Label({style: {backgroundColor: cor, padding: '7px', margin: '0 6px 0 0', border: '1px solid #ccc'}}));
    row.add(ui.Label(texto, {fontSize: '11px', margin: '1px 0'}));
    return row;
};

var legenda = ui.Panel([
    legendaTitulo,
    mkLegRow('#FF0000', 'Predição modelo (> 0.5)'),
    mkLegRow('#FFD700', 'Buffer 5 km (regiões)'),
    mkLegRow('#1a9641', 'PVI – vegetação / baixo'),
    mkLegRow('#d7191c', 'PVI – painel solar / alto'),
]);

// --- Separador ---
var sep = ui.Label('─────────────────────', {color: '#ccc', fontSize: '11px', margin: '6px 0'});

// --- Montagem do painel ---
var painel = ui.Panel({
    widgets: [
        titulo, subtitulo,
        labelAno,  selectAno,
        labelSem,  selectSem,
        labelReg,  selectReg,
        btnAplicar,
        statusLabel,
        sep,
        legenda
    ],
    style: {
        width: '240px',
        padding: '12px',
        backgroundColor: '#f9f9f9'
    }
});

ui.root.insert(0, painel);

// ============================================================
// 6. RENDERIZAÇÃO INICIAL
// ============================================================
Map.setOptions('SATELLITE');
renderizar(2024, '2º Semestre', 'Todas');
statusLabel.setValue('Exibindo: 2024 | 2º Semestre | Região: Todas');

// var regions_excluidos = ee.FeatureCollection(excluir_regions);
// var name_export = 'poligons_exclusion_comision_16_05_2026';
// processoExportar(regions_excluidos, name_export);

// var regions_complementar = ee.FeatureCollection(complementar);
// var name_export = 'poligons_complementar_omision_16_05_2026';
// processoExportar(regions_complementar, name_export);


// var regs_complementar = shp_polFV_in.merge(ee.FeatureCollection(complementar));
// print(" ver regions complemenates ", regs_complementar);
// name_export = 'polygons_base_paneis_fotovoltaicos_16_05_2026';
// processoExportar(regs_complementar, name_export);
// Map.addLayer(regs_complementar, {'colo': 'blue'}, 'regs_complementar');

var points_coleta = ee.FeatureCollection(ver).merge(bom_predict);
// name_export = 'pontos_areas_DB_16_05_2026';
// processoExportar(points_coleta, name_export);

var new_area_sample = col_usinaFV.filterBounds(points_coleta);
print("metada data ", new_area_sample);
var ampliar_vision = ee.FeatureCollection(ampliar_visao).map(
                                function(feat){
                                    return feat.bounds().set(
                                        'ORIG_FID', 1,
                                        'Shape_Area', ee.Number(feat.geometry().area()).divide(10000),
                                        'Shape_Leng', ee.Number(feat.geometry().perimeter()).divide(10000)
                                    );
                                })
Map.addLayer(new_area_sample, {color: 'red'}, 'new_area');
new_area_sample = col_usinaFV.merge(ampliar_vision);
print("new area samples ", new_area_sample);

Map.addLayer(new_area_sample, {color: 'yellow'}, 'new_area +');
var name_export = 'shp_area_fotovoltaic_samples_update_16_05_2026';
processoExportar(new_area_sample, name_export);