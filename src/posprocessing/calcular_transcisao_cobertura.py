"""Change-of-cover statistics under solar (PV, class 75) and wind (class 91) patches.

Idea (guided by area_fv_por_ano_bioma_col11.js):
  - The 2025 layer tells us WHERE energy infrastructure is (classes 75 / 91).
  - The 2013 layer tells us WHAT was there before (savanna, forest, pasture...).
  - Intersecting the 2025 energy mask with the 2013 classification therefore
    yields, per prior cover class, the area that was later converted to solar
    or wind. That converted area is our "transformation" estimate.

We compute, restricted to the Caatinga biome, area (ha) of each 2013 cover class
lying under 2025 PV/wind patches, broken down by state. Everything runs
server-side (reduceRegions over the Caatinga states) and the single flat table is
exported to Google Drive via a batch task -- no per-state local files.

Because the states FeatureCollection is already filtered to CD_Bioma == 2, a
state that only partially overlaps the biome (e.g. Minas Gerais) contributes only
its Caatinga portion.

Run:  .venv/bin/python calcular_transcisao_cobertura.py
      The task runs asynchronously; watch it at
      https://code.earthengine.google.com/tasks
Auth: run `earthengine authenticate` once if needed.
"""

import os
import ee
import sys
from pathlib import Path
# ---------------------------------------------------------------- config
pathparent = str(Path(os.getcwd()).parents[0])
sys.path.append(pathparent)
from configure_account_projects_ee import get_current_account

projAccount = get_current_account()
print(f"Projeto selecionado >>> {projAccount} <<<")



COVERAGE_ASSET = (
    "projects/mapbiomas-public/assets/brazil/lulc/collection11/"
    "mapbiomas_brazil_collection11_coverage_v3"
)
YEAR_BEFORE = 2013
YEAR_AFTER = 2025
ENERGY_CLASSES = {75: "Fotovoltaica", 91: "Eolica"}
SCALE = 30
BIOME = "Caatinga"

# Biome+state boundary: one feature per (biome, state) intersection.
# CD_Bioma == 2 is Caatinga; NM_ESTADO holds the state name.
ESTADOS_FC = os.environ.get(
    "ESTADOS_FC",
    "projects/mapbiomas-workspace/AUXILIAR/biomas_IBGE_estados_2016_old",
)
CD_BIOMA = int(os.environ.get("CD_BIOMA", "2"))  # 2 = Caatinga
UF_PROPERTY = os.environ.get("UF_PROPERTY", "NM_ESTADO")

# Google Drive destination for the exported table.
DRIVE_FOLDER = os.environ.get("DRIVE_FOLDER", "MAPBIOMAS_ENERGIA")
OUT_PREFIX = "mudanca_cobertura_energia"

# MapBiomas Collection 11 legend (main classes) for readable output.
LEGENDA = {
    1: "Floresta", 3: "Formação Florestal", 4: "Formação Savânica",
    5: "Mangue", 6: "Floresta Alagável", 49: "Restinga Arbórea",
    10: "Formação Natural não Florestal", 11: "Campo Alagado/Área Pantanosa",
    12: "Formação Campestre", 32: "Apicum", 29: "Afloramento Rochoso",
    50: "Restinga Herbácea", 13: "Outra Formação não Florestal",
    14: "Agropecuária", 15: "Pastagem", 18: "Agricultura", 19: "Lavoura Temporária",
    39: "Soja", 20: "Cana", 40: "Arroz", 62: "Algodão", 41: "Outra Lavoura Temporária",
    36: "Lavoura Perene", 46: "Café", 47: "Citrus", 35: "Dendê",
    48: "Outra Lavoura Perene", 9: "Silvicultura", 21: "Mosaico de Usos",
    22: "Área não Vegetada", 23: "Praia/Duna/Areal", 24: "Área Urbanizada",
    30: "Mineração", 25: "Outra Área não Vegetada",
    26: "Corpo d'Água", 33: "Rio/Lago/Oceano", 31: "Aquicultura",
    27: "Não Observado", 75: "Usina Fotovoltaica", 91: "Parque Eólico",
}


def init_ee():
    try:
        ee.Initialize(project=projAccount)
        print('Earth Engine inicializado com sucesso!')
    except Exception as e:
        print("Erro de Inicialização:", e)
        raise


def caatinga_estados_fc():
    """FeatureCollection with one feature per Caatinga state."""
    return ee.FeatureCollection(ESTADOS_FC).filter(
        ee.Filter.eq("CD_Bioma", CD_BIOMA)
    )


def linhas_energia(coverage, estados_fc, code, nome, legenda_ee):
    """Server-side FeatureCollection of (estado, energia, classe_2013, area_ha).

    One reduceRegions over every Caatinga state, grouped by the 2013 class, then
    flattened so each output feature is a single (state, prior-class) area.
    """
    band_before = f"classification_{YEAR_BEFORE}"
    band_after = f"classification_{YEAR_AFTER}"
    before = coverage.select(band_before)
    energy_mask = coverage.select(band_after).eq(code)

    # pixelArea (band 0) grouped by 2013 class (band 1), only under energy patches.
    img = ee.Image.pixelArea().addBands(before).updateMask(energy_mask)
    reducer = ee.Reducer.sum().group(groupField=1, groupName="class_2013")
    reduced = img.reduceRegions(
        collection=estados_fc,
        reducer=reducer,
        scale=SCALE,
        tileScale=4,
    )

    def por_estado(feat):
        uf = feat.get(UF_PROPERTY)
        grupos = ee.List(feat.get("groups"))

        def por_grupo(g):
            g = ee.Dictionary(g)
            cls = ee.Number(g.get("class_2013")).toInt()
            cls_str = cls.format()
            return ee.Feature(None, {
                "estado": uf,
                "energia": nome,
                "energia_classe": code,
                "cobertura_2013_classe": cls,
                "cobertura_2013_nome": legenda_ee.get(cls_str, cls_str),
                "area_ha": ee.Number(g.get("sum")).divide(1e4),
            })

        return ee.FeatureCollection(grupos.map(por_grupo))

    return reduced.map(por_estado).flatten()


def main():
    init_ee()
    coverage = ee.Image(COVERAGE_ASSET)
    estados_fc = caatinga_estados_fc()
    # Server-side lookup needs string keys (EE dictionaries are keyed by string).
    legenda_ee = ee.Dictionary({str(k): v for k, v in LEGENDA.items()})

    partes = [
        linhas_energia(coverage, estados_fc, code, nome, legenda_ee)
        for code, nome in ENERGY_CLASSES.items()
    ]
    tabela = ee.FeatureCollection(partes).flatten()

    selectors = [
        "estado", "energia", "energia_classe",
        "cobertura_2013_classe", "cobertura_2013_nome", "area_ha",
    ]
    task = ee.batch.Export.table.toDrive(
        collection=tabela,
        description=OUT_PREFIX,
        folder=DRIVE_FOLDER,
        fileNamePrefix=OUT_PREFIX,
        fileFormat="CSV",
        selectors=selectors,
    )
    task.start()
    print(f"Task iniciada -> Drive/{DRIVE_FOLDER}/{OUT_PREFIX}.csv")
    print(f"  id da task: {task.id}")
    print("  acompanhe em https://code.earthengine.google.com/tasks")


if __name__ == "__main__":
    main()
