// ============================================================
// Área de Painéis Fotovoltaicos por Ano e Bioma — MapBiomas Col. 11
// ============================================================
// Calcula a área (m², ha, km²) de pixels fotovoltaicos (valor == 1)
// por ano e por bioma, e exporta uma tabela CSV para o Google Drive.
//
// Entrada:
//   ImageCollection: projects/mapbiomas-brazil/assets/LAND-COVER/
//                    COLLECTION-11/RENEWABLE-ENERGY/solar-panels
//   Biomas: projects/mapbiomas-workspace/AUXILIAR/bioma_2025_e250k_5kbuffer
//
// Saída:
//   CSV no Drive → pasta MapBiomas_exports/
//   Colunas: year | Bioma | area_m2 | area_ha | area_km2
// ============================================================

// ── Assets ────────────────────────────────────────────────────

var ic = ee.ImageCollection(
  'projects/mapbiomas-brazil/assets/LAND-COVER/COLLECTION-11/RENEWABLE-ENERGY/solar-panels'
);
print("show metadado de Fotovoltaicas ", ic);

var lista_biomas = ["Amazônia", "Caatinga", "Cerrado", "Mata Atlântica", "Pampa", "Pantanal"];
var biomas = ee.FeatureCollection(
  'projects/mapbiomas-workspace/AUXILIAR/bioma_2025_e250k_5kbuffer'
).map(function(feat){return feat.set('id_cod', 1)});
print("show metadado de Biomas ", biomas);



// Resolução nativa dos mapas MapBiomas (metros)
var SCALE = 30;

// ── Diagnóstico ───────────────────────────────────────────────
// Rodar esta seção primeiro para confirmar:
//   1) o nome da propriedade de ano na ImageCollection
//   2) o nome da coluna de bioma na FeatureCollection (ex: 'Bioma', 'bioma', 'NM_BIOMA')
// Ajustar BIOMA_COL e YEAR_PROP abaixo conforme necessário.

print('IC — tamanho (nº de imagens):', ic.size());
print('IC — 1ª imagem (ver propriedades):', ic.first());
print('Biomas — 1ª feature (ver propriedades):', biomas.first());

// ── Configuração ──────────────────────────────────────────────
// Nome da propriedade que identifica o ano em cada imagem da IC
var YEAR_PROP = 'year';

// ── Cálculo de área por ano e bioma ──────────────────────────

var resultList = ic.toList(ic.size()).map(function(img) {
    img = ee.Image(img);
    var year = img.get('year');  // propriedade 'year' existe; img.date() falha pois não há system:time_start

    var area_biomas_yy = ee.List(lista_biomas).map(function(name_bioma) {
        var bioma_select = biomas.filter(ee.Filter.eq('NAME', name_bioma));
        var bioma_mask   = bioma_select.reduceToImage(['id_cod'], ee.Reducer.first());

        var classified = img.select('b1').updateMask(bioma_mask);

        var areaImg = classified.eq(1)
                                .multiply(ee.Image.pixelArea())
                                .rename('area_m2');

        var perBioma = areaImg.reduceRegions({
            collection: bioma_select,
            reducer:    ee.Reducer.sum(),
            scale:      SCALE,
            tileScale:  4,
        });

        return perBioma.map(function(feat) {
            var area_m2 = ee.Number(feat.get('sum'));
            return feat
                .set('year',    year)
                .set('Bioma',   name_bioma)
                .set('area_ha', area_m2.divide(1e4));
        }).toList(10);  // fix 2: FeatureCollection → List
    });

    return area_biomas_yy.flatten();
});

var result = ee.FeatureCollection(resultList.flatten());  // fix 3

print('Resultado — primeiras 12 linhas:', result.limit(12));


// ── Export para Google Drive ──────────────────────────────────

var exportCols = ['year', 'Bioma', 'area_ha'];

Export.table.toDrive({
  collection:     result,
  description:    'solar_panels_area_by_year_biome_2025',
  folder:         'MapBiomas_exports',
  fileNamePrefix: 'solar_panels_area_by_year_biome_2025',
  fileFormat:     'CSV',
  selectors:      exportCols,
});
