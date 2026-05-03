import os 
from tqdm import tqdm
import shutil
param = {
    'rutaInputpatches': '/home/superuser/Dados/mapbiomas/mapping_areas_eolicas_fotovoltaicas/src/dados/patchs_FV_g2',
    'file_fails': '/home/superuser/Dados/mapbiomas/mapping_areas_eolicas_fotovoltaicas/src/dados/reg_faltantes.txt',
    'rutaOuputPatches': '/home/superuser/Dados/mapbiomas/mapping_areas_eolicas_fotovoltaicas/src/dados/toMove'
}

lstfilesfails = []
arqTXT = open(param['file_fails'])
cc = 0
for line in tqdm(arqTXT):
    lstfilesfails.append(line[:-1]) 
    if cc < 5:
        print(line[:-1])
    cc += 1

lst_folders = os.listdir(param['rutaInputpatches'])
lstFileMove = []
cc = 0
for nfile in tqdm(lst_folders):
    if nfile not in lstfilesfails:
        lstFileMove.append(nfile)
    if cc < 5:
        print(nfile)
    cc += 1 
print(f" we have {len(lstFileMove)} to move to other folder ")    
for nfile in lstFileMove:
    source = os.path.join(param['rutaInputpatches'], nfile)
    destination = os.path.join(param['rutaOuputPatches'], nfile)    
    shutil.copyfile(source, destination)
    print("moving >>> ", nfile)
