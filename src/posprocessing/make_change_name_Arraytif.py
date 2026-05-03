import os 
import glob

path_base = '/home/superuser/Dados/mapbiomas/mapping_areas_eolicas_fotovoltaicas/src/dados/array_pred_tif_g2'
path_output = '/home/superuser/Dados/mapbiomas/mapping_areas_eolicas_fotovoltaicas/src/dados/array_pred_tif_g2_corr/'

for nyear in range(2015, 2026):
    print("=============================================================")
    lstpathTIF = glob.glob(os.path.join(path_base, str(nyear)) + '/*.tif')
    print(f" loading {len(lstpathTIF)} tif in >>>>> \n >> ", os.path.join(path_base, str(nyear)))
    pathDest = os.path.join(path_output, str(nyear))
    if not os.path.exists(pathDest):
        os.makedirs(pathDest)
    print("=============================================================")
    for npath in lstpathTIF[:]:
        # print(npath)
        name_Array = npath.split("/")[-1]
        print(name_Array)
        partes = name_Array.split("_")
        # print(partes)
        if len(partes) < 6:
            new_name = f"reg_{partes[1]}_{partes[2]}_10_pred.tif"            
        else:
            new_name = f"reg_{partes[1]}_{partes[2]}_{partes[4]}_pred.tif"
        print(f"change <{name_Array}> ... \n >>>> by new name <{new_name}>")
        os.rename(npath, os.path.join(pathDest, new_name))