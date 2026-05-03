/**
 * Visualização Integrada: Planet NICFI + Índices + MapBiomas 30m (Col 10)
 *
 * Objetivo: Detectar e visualizar painéis fotovoltaicos usando índices espectrais
 * derivados de imagens Planet NICFI (bandas: B, G, R, N).
 *
 * Escala de armazenamento (Int16):
 *   Índices normalizados [-1,1]  → .add(1).multiply(10000) → faixa [0, 20000]
 *   PVSI (razão RGB/NIR)         → .multiply(1000)         → faixa [0,  ~2000]
 *   PGI  (percentual ~33-39)     → .multiply(100)          → faixa [3300, 3900]
 */

// ============================================================
// 1. ASSETS E COLEÇÕES
// ============================================================
var asset_usinaFV = 'users/mapbiomascaatinga04/atualiz_buffer_fotovoltaic_5km';
var asset_NICFI   = 'projects/planet-nicfi/assets/basemaps/americas';
var asset_new_fv  = 'projects/mapbiomas-workspace/AMOSTRAS/col10/CAATINGA/solar-panel-br-30m_2016_2024_v2';

var col_usinaFV  = ee.FeatureCollection(asset_usinaFV);
var mosaic30m_fv = ee.Image(asset_new_fv);

// ============================================================
// 2. FUNÇÕES DE CÁLCULO DE ÍNDICES
// ============================================================
/**
 * Calcula oito índices espectrais e retorna imagem com as bandas
 * originais (B, G, R, N) + índices, todos em Int16.
 */
var calculateIndiceAllinOne = function(img) {
    var b = img.select(['B', 'G', 'R', 'N']);

    // IIA – Índice de Absorção/Água: baixos valores indicam superfícies absorventes
    var iia = img.expression('float((G - 4*N) / (G + 4*N))', {
        G: b.select('G'), N: b.select('N')
    }).add(1).multiply(10000).rename('iia');

    // EVI – Enhanced Vegetation Index
    var evi = img.expression('float(2.4 * (N - R) / (1 + N + R))', {
        N: b.select('N'), R: b.select('R')
    }).add(1).multiply(10000).rename('evi');

    // PVI – Photovoltaic Index (contraste Vermelho/NIR)
    var pvi = img.expression('float((R - N) / (R + N))', {
        R: b.select('R'), N: b.select('N')
    }).add(1).multiply(10000).rename('pvi');

    // PVPI – Photovoltaic Panel Index (Verde/Azul)
    var pvpi = img.expression('float((G - B) / (G + B))', {
        G: b.select('G'), B: b.select('B')
    }).add(1).multiply(10000).rename('pvpi');

    // PGI – Plastic Greenhouse Index (domínio do verde em %)
    var pgi = img.expression('float((100.0 * G) / (R + G + B))', {
        R: b.select('R'), G: b.select('G'), B: b.select('B')
    }).multiply(100).rename('pgi');

    // PVSI – Photovoltaic Spectral Index (razão RGB/NIR para silício)
    var pvsi = img.expression('float(((R + G + B) / 3.0) / (N + 0.1))', {
        R: b.select('R'), G: b.select('G'), B: b.select('B'), N: b.select('N')
    }).multiply(1000).rename('pvsi');

    // RI – Redness Index (presença de solo exposto)
    var ri = img.expression('float(2.4 * (R - G) / (R + G))', {
        R: b.select('R'), G: b.select('G')
    }).add(1).multiply(10000).rename('ri');

    // SHAPE – Índice de Estrutura (contraste R vs G-B)
    var shape = img.expression('float((2*R - G - B) / (G - B + 0.001))', {
        R: b.select('R'), G: b.select('G'), B: b.select('B')
    }).add(1).multiply(10000).rename('shape');

    return img.select(['B', 'G', 'R', 'N']).toInt16()
        .addBands([
            iia.toInt16(), evi.toInt16(), pvi.toInt16(),
            pvpi.toInt16(), pgi.toInt16(), pvsi.toInt16(),
            ri.toInt16(), shape.toInt16()
        ]);
};

// ============================================================
// 3. PROCESSAMENTO DO MOSAICO PLANET
// ============================================================
var planetMosaic = ee.ImageCollection(asset_NICFI)
    .filter(ee.Filter.date('2024-07-01', '2024-12-31'))  // 2º semestre 2024
    .median()
    .clip(col_usinaFV);

var planetWithIndices = calculateIndiceAllinOne(planetMosaic);

// ============================================================
// 4. PARÂMETROS DE VISUALIZAÇÃO (valores em escala Int16)
// ============================================================
// Imagem base RGB
var visRGB = {bands: ['R', 'G', 'B'], min: 20, max: 3500, gamma: 1.8};

// Índices de detecção de painéis
// PVSI: raw [0.19, 0.69] × 1000      → Int16 [190, 690]
var visPVSI = {min: 111,   max: 715,   palette: ['black', 'yellow', 'red']};
// PVI:  raw [-0.6, 0.15] +1 × 10000  → Int16 [4000, 11500]
var visPVI  = {min: 2066,  max: 9110, palette: ['blue', 'white', 'red']};
// PGI:  raw [33, 39] × 100           → Int16 [3300, 3900]
var visPGI  = {min: 2957,  max: 4513,  palette: ['blue', 'white', 'red']};
// PVPI: raw [0.28, 0.55] +1 × 10000  → Int16 [12800, 15500]
var visPVPI = {min: 10950, max: 13950, palette: ['blue', 'white', 'red']};

// Índices de contexto ambiental
var visIIA   = {min: 1183,  max: 1572, palette: ['#000033', '#0000ff', '#00ffff', '#ffffff']};
var visEVI   = {min: 13700, max: 23950, palette: ['#ffffff', '#f7fcb9', '#addd8e', '#31a354', '#00441b']};
var visRI    = {min: 10867,  max: 15170, palette: ['#fff7bc', '#fec44f', '#d95f0e', '#662506']};
var visSHAPE = {min: 23530,  max: 32760, palette: ['#ece7f2', '#a6bddb', '#d8b365', '#5ab4ac']};

// ============================================================
// 5. VISUALIZAÇÃO NO MAPA
// ============================================================
Map.centerObject(col_usinaFV, 14);

// Imagem base
Map.addLayer(planetMosaic, visRGB, '1. Planet RGB (Real)', false);

// Índices de detecção de painéis (Planet)
Map.addLayer(planetWithIndices.select('pvsi'), visPVSI, '2. PVSI – Silício',  true);
Map.addLayer(planetWithIndices.select('pvi'),  visPVI,  '3. PVI – Painel',    false);
Map.addLayer(planetWithIndices.select('pgi'),  visPGI,  '4. PGI – Plástico',  false);
Map.addLayer(planetWithIndices.select('pvpi'), visPVPI, '5. PVPI – Painel',   false);

// Índices de contexto ambiental
Map.addLayer(planetWithIndices.select('iia'),   visIIA,   '6. IIA – Água/Absorção',   false);
Map.addLayer(planetWithIndices.select('evi'),   visEVI,   '7. EVI – Vegetação',        false);
Map.addLayer(planetWithIndices.select('ri'),    visRI,    '8. RI – Solo/Vermelhidão',  false);
Map.addLayer(planetWithIndices.select('shape'), visSHAPE, '9. SHAPE – Solo/Estrutura', false);

// Vetores e referência
Map.addLayer(col_usinaFV, {color: '00FFFF'}, '10. Pontos Usinas FV');
Map.addLayer(
    mosaic30m_fv.select('Panel_2024').selfMask(),
    {palette: ['#FF0000']},
    '11. Mapeamento FV 30m (Referência)',
    true
);

// ============================================================
// 6. DEBUG / INSPEÇÃO
// ============================================================
print('Metadata 30 metros:', mosaic30m_fv);
print('Bandas Planet com Índices:', planetWithIndices.bandNames());
