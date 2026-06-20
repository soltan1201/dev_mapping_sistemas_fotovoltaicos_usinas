var asset_grade = "projects/nexgenmap/SAD_MapBiomas/DL/SHP_grades_BR_35pathces_AllBrV3";
var asset_photovoltaica = 'projects/mapbiomas-arida/fotovoltaic_rural';
var colection_fv = ee.FeatureCollection(asset_photovoltaica);

var grid_shp = ee.FeatureCollection(asset_grade);
var grades = grid_shp.filterBounds(colection_fv);
grades = grades.map(
            function(feat){
                var quant_size = colection_fv.filterBounds(feat.geometry()).size();
                return feat.set('quantidade', quant_size);
            })

print(" colection_fv ", colection_fv.size());
print("show metadata ", colection_fv.limit(5));            
print("show grades com quantidade ", grades.limit(5));
Map.addLayer(colection_fv, {color: 'red'}, 'Fotovoltaica');
// Create an empty image into which to paint the features, cast to byte.
// Paint all the polygon edges with the same number and width, display.
var outline = ee.Image().byte().paint(grades, 1, 2);
Map.addLayer(outline, {palette: '000000'}, 'edges');

var name_export = "grade_fotovoltaic_rural_2026";
Export.table.toAsset({
    collection: grades, 
    description: name_export, 
    assetId: 'projects/mapbiomas-arida/' + name_export
})