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
 * Índices de solo exposto via Sentinel-2 SR (B2=B, B3=G, B4=R, B8=N, B11=S1, B12=S2).
 * Valores calculados sobre DN bruto (escala 0-10000).
 * Retorna imagem com bandas: BSI, MBI, DBSI, NDSI_SOIL.
 */
var calcularIndicesSolo = function(s2) {
  var bsi = s2.expression(
    '((S1 + R) - (N + B)) / ((S1 + R) + (N + B))', {
    S1: s2.select('B11'), R: s2.select('B4'),
    N:  s2.select('B8'),  B: s2.select('B2')
  }).rename('BSI');

  var mbi = s2.expression(
    '((S1 - S2 - N) / (S1 + S2 + N)) + 0.5', {
    S1: s2.select('B11'), S2: s2.select('B12'),
    N:  s2.select('B8')
  }).rename('MBI');

  var dbsi = s2.expression(
    '((S1 - G) / (S1 + G)) - ((N - R) / (N + R))', {
    S1: s2.select('B11'), G: s2.select('B3'),
    N:  s2.select('B8'),  R: s2.select('B4')
  }).rename('DBSI');

  var ndsi_soil = s2.expression(
    '(S2 - G) / (S2 + G)', {
    S2: s2.select('B12'), G: s2.select('B3')
  }).rename('NDSI_SOIL');

  return ee.Image([bsi, mbi, dbsi, ndsi_soil]);
};

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

    // OSAVI – Optimized Soil-Adjusted Vegetation Index
    var osavi = img.expression('float((N - R) / (0.16 + N + R))', {
        N: b.select('N'), R: b.select('R')
    }).add(1).multiply(10000).rename('osavi');

    // GNDVI – Green Normalized Difference Vegetation Index
    var gndvi = img.normalizedDifference(['N', 'G'])
        .add(1).multiply(10000).rename('gndvi');

    // RI – Redness Index (presença de solo exposto)
    var ri = img.expression('float(2.4 * (R - G) / (R + G))', {
        R: b.select('R'), G: b.select('G')
    }).add(1).multiply(10000).rename('ri');

    // SHAPE – Índice de Estrutura (contraste R vs G-B)
    var shape = img.expression('float((2*R - G - B) / (G - B + 0.001))', {
        R: b.select('R'), G: b.select('G'), B: b.select('B')
    }).add(1).multiply(10000).rename('shape');

    // NDWI – Normalized Difference Water Index (Green − NIR) / (Green + NIR)
    var ndwi = img.normalizedDifference(['G', 'N'])
        .add(1).multiply(10000).rename('ndwi');

    return img.select(['B', 'G', 'R', 'N']).toInt16()
        .addBands([
            iia.toInt16(), evi.toInt16(), pvi.toInt16(),
            pvpi.toInt16(), pgi.toInt16(), pvsi.toInt16(),
            osavi.toInt16(), gndvi.toInt16(),
            ri.toInt16(), shape.toInt16(), ndwi.toInt16()
        ]);
};

// ============================================================
// 3. PROCESSAMENTO DOS MOSAICOS
// ============================================================
var planetMosaic = ee.ImageCollection(asset_NICFI)
    .filter(ee.Filter.date('2024-07-01', '2024-12-31'))  // 2º semestre 2024
    .median()
    .clip(col_usinaFV);

var planetWithIndices = calculateIndiceAllinOne(planetMosaic);

// Sentinel-2 SR — mesmo período, nuvens < 20 %
var s2Mosaic = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filter(ee.Filter.date('2024-07-01', '2024-12-31'))
    .filter(ee.Filter.bounds(col_usinaFV))
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .median()
    .clip(col_usinaFV);

var soloIndices = calcularIndicesSolo(s2Mosaic);

// ── Fusão Planet + S2 SWIR2 a 5 m ──────────────────────────────────
// B12 (SWIR2, 20 m nativo) reprojetado para 5 m via bilinear,
// alinhado à projeção do mosaico Planet.
var proj5m = planetMosaic.projection().atScale(4.7);

var s2swir2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filter(ee.Filter.date('2024-07-01', '2024-12-31'))
    .filter(ee.Filter.bounds(col_usinaFV))
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    .select('B12')
    .median()
    .resample('bilinear')   // interpola 20 m → 5 m suavemente
    .reproject(proj5m)      // força o grid de 5 m do Planet
    .clip(col_usinaFV)
    .rename('SWIR2');

// Fusão: Planet (G, R, N, B…) + SWIR2 reescalado
var planetFused = planetMosaic.addBands(s2swir2);

// NDSoiI = (SWIR2 − G) / (SWIR2 + G)   — ambas as bandas em DN 0-10000
var ndsiSoilFused = planetFused.expression(
    '(S2 - G) / (S2 + G)', {
      S2: planetFused.select('SWIR2'),
      G:  planetFused.select('G')
    }).rename('NDSI_SOIL_5m');

// ============================================================
// 4. PARÂMETROS DE VISUALIZAÇÃO (valores em escala Int16)
// ============================================================
// Imagem base RGB
var visRGB = {bands: ['R', 'G', 'B'], min: 20, max: 3500, gamma: 1.8};

// Índices de detecção de painéis
var PAL_FV = ["662506","fec44f","ffffff","006837"];
// PVSI: raw [0.19, 0.69] × 1000      → Int16 [190, 690]
var visPVSI = {min: 111,   max: 505,   palette: PAL_FV};
// PVI:  raw [-0.6, 0.15] +1 × 10000  → Int16 [4000, 11500]
var visPVI  = {min: 1993,  max: 7876, palette: PAL_FV};
// PGI:  raw [33, 39] × 100           → Int16 [3300, 3900]
var visPGI  = {min: 2861,  max: 4757,  palette: PAL_FV};
// PVPI: raw [0.28, 0.55] +1 × 10000  → Int16 [12800, 15500]
var visPVPI = {min: 11711, max: 14052, palette: PAL_FV};

// Índices de contexto ambiental
var visIIA   = {min: 738,  max: 2257,  palette: ['#000033', '#0000ff', '#00ffff', '#ffffff']};
var visEVI   = {min: 15096, max: 29212, palette: ['#ffffff', '#f7fcb9', '#addd8e', '#31a354', '#00441b']};
// OSAVI (N−R)/(0.16+N+R) — versão do NDVI com ajuste de solo fixo (L=0.16), mais estável em regiões semiáridas com solo exposto entre painéis
// OSAVI: raw [-1,1] +1 ×10000 → [0,20000]; vegetal alto > 15000
var visOSAVI = {min: 12123,  max: 18006, palette: ['#8B4513', '#ffffbf', '#006837']};
// GNDVI (N−G)/(N+G) — usa Green em vez de Red, mais sensível a variações de clorofila em dosséis densos e distingue melhor vegetação de painéis que o NDVI clássico
// GNDVI: raw [-1,1] +1 ×10000 → [0,20000]; mais sensível a clorofila que NDVI
var visGNDVI = {min: 9000,  max: 17000, palette: ['#8B4513', '#ffffbf', '#006837']};
var visRI    = {min: 4191, max: 18159, palette: ['#fff7bc', '#fec44f', '#d95f0e', '#662506']};
var visSHAPE = {min: 8676, max: 30090, palette: ['#ece7f2', '#a6bddb', '#d8b365', '#5ab4ac']};
// NDWI: raw [-1,1] +1 ×10000 → Int16 [0,20000]; água > 10000, solo/veg < 10000
var visNDWI  = {min: 2954,  max: 6465, palette: ['#000033', '#0000ff', '#00ffff', '#ffffff']};

// Índices de solo exposto — Sentinel-2 (paleta única: verde → creme → marrom)
var PAL_SOLO       = ["#fff7bc","#fec44f","#d95f0e","#662506"];
var visBSI         = {min: -0.27, max:  0.37, palette: PAL_SOLO};
var visMBI         = {min:  0.06, max:  0.41, palette: PAL_SOLO};
var visDBSI        = {min: -0.33, max:  0.54, palette: PAL_SOLO};
var visNDSoil      = {min: 0.17, max:  0.65, palette: PAL_SOLO};
// NDSoiI fusionado Planet 5 m + S2 SWIR2
var visNDSoilFused = {min: -0.2, max:  0.4, palette: PAL_SOLO};
// SWIR2 bruto (DN 0-10000) — útil para inspeção visual
var visSWIR2       = {min: 200,  max: 3000,  palette: ['#000000','#ffffbf','#c86400']};

// ============================================================
// 5. VISUALIZAÇÃO NO MAPA
// ============================================================
// Map.centerObject(col_usinaFV, 14);

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
Map.addLayer(planetWithIndices.select('osavi'), visOSAVI, '7b. OSAVI – Veg. (solo adj.)', false);
Map.addLayer(planetWithIndices.select('gndvi'), visGNDVI, '7c. GNDVI – Veg. (clorofila)', false);
Map.addLayer(planetWithIndices.select('ri'),    visRI,    '8. RI – Solo/Vermelhidão',  false);
Map.addLayer(planetWithIndices.select('shape'), visSHAPE, '9. SHAPE – Solo/Estrutura', false);
Map.addLayer(planetWithIndices.select('ndwi'),  visNDWI,  '9b. NDWI – Água/Superfície', false);

// Índices de solo exposto — Sentinel-2 SR (20 m nativo)
Map.addLayer(soloIndices.select('BSI'),       visBSI,    '10. BSI – Solo Exposto (S2)',      false);
Map.addLayer(soloIndices.select('MBI'),       visMBI,    '11. MBI – Solo Exposto Mod. (S2)', false);
Map.addLayer(soloIndices.select('DBSI'),      visDBSI,   '12. DBSI – Solo Seco (S2)',        false);
Map.addLayer(soloIndices.select('NDSI_SOIL'), visNDSoil, '13. NDSoiI – Solo (S2 20m)',      false);

// Fusão Planet 5 m + S2 SWIR2 reescalado
Map.addLayer(s2swir2,       visSWIR2,       '14. SWIR2 S2 → 5 m (bruto)',          false);
Map.addLayer(ndsiSoilFused, visNDSoilFused, '15. NDSoiI fusionado Planet+S2 (5m)', true);

// Vetores e referência
Map.addLayer(col_usinaFV, {color: '00FFFF'}, '14. Pontos Usinas FV');
Map.addLayer(
    mosaic30m_fv.select('Panel_2024').selfMask(),
    {palette: ['#FF0000']},
    '15. Mapeamento FV 30m (Referência)',
    true
);

// ============================================================
// 6. DEBUG / INSPEÇÃO
// ============================================================
print('Metadata 30 metros:', mosaic30m_fv);
print('Bandas Planet com Índices:', planetWithIndices.bandNames());
print('Bandas Solo (S2):', soloIndices.bandNames());
print('Projeção Planet 5 m:', proj5m);
print('NDSoiI fusionado:', ndsiSoilFused);
