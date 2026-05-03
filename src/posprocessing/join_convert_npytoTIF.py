
# from osgeo import gdal
import rasterio
from rasterio.transform import from_origin
from os import system
# import random
# import math
import json
import os
import sys
import numpy as np
from tqdm import tqdm
import gc 

def save_array_to_TIF(name_array, img_array, mytransform, yyear):
    # Definindo os metadados
    # Adapte os valores para o seu caso específico
    # transform = from_origin(mytransform)  # Definindo a transformação (origem e resolução)
    crs = 'EPSG:4326'  # Definindo o sistema de coordenadas
    crs = "EPSG:3857"
    name_img= os.path.join(os.path.join(param['rutaOutpatches'], str(yyear)), f"{name_array}_pred.tif")
    # Abrindo um novo dataset TIFF para escrita
    with rasterio.open(name_img, 'w',
                    driver='GTiff',
                    height=img_array.shape[0],
                    width=img_array.shape[1],
                    count=1,  # Número de bandas
                    dtype=img_array.dtype,
                    crs=crs,
                    transform= mytransform) as dst:
        dst.write(img_array, 1)  # Escrevendo os dados na primeira banda

    print("Imagem TIFF gerada com sucesso!")


def join_patchs (patch_name, lstnameArr, iBuffer, dimPatch, myear):

    dictMetaD = {}
    idCode = patch_name.split("_")[1]
    with open(param['rutaJson_atrib'] + "/" + patch_name + '.json') as json_file:
        dictMetaD = json.load(json_file)
    print("  transform ", dictMetaD["transform"])
    for vkey, vlist in dictMetaD.items():
        print(f" {vkey} ", vlist)
    
    sideX = dictMetaD["xsize"]
    sideY = dictMetaD["ysize"]
    print(f" sideX {sideX}   |  sideY {sideY}")
    quantX = int(sideX / dimPatch)
    restoX = int(sideX % dimPatch)
    quantY = int(sideY / dimPatch)
    restoY = int(sideY % dimPatch)

    print(f" quantX {quantX} |  restoX  {restoX} ")
    print(f" quantY {quantY} |  restoY  {restoY} ")
    print(f" {len(lstnameArr)}  >>>> { quantX * quantY}")

    if (restoX - 10) > 0  or (restoY - 10) > 0:
         iBuffer = 0

    lstArray = []
    for cc in range(len(lstnameArr)):
        pathname_array = param['rutaInputImg'] + "/" + patch_name +  f'_g{cc}.npy' # _{myear}
        narray0 = np.load(pathname_array)
        lstArray.append(narray0[:,:])

    print("tamaño do array ", len(lstArray))
    lstCoorArray = []
    # if sideX > dimPatch or sideY > dimPatch:
    if iBuffer == 0:
        print("iterando por corte")
        cc = 0
        posY = (quantY * dimPatch) - sideY 
        posX = (quantX * dimPatch) - sideX 
        lstGeral = []
        for xx in range(0, quantX + 1)[:]:
            if ((xx + 1) * dimPatch ) <  sideX:
                lst_tmp = []
                for yy in range(0, quantY + 1):
                    print(f"add patch {cc} com Y {(yy + 1) * dimPatch + 1}")
                    if ((yy + 1)* dimPatch + 1) <  sideY:
                        print(lstArray[cc].shape)
                        lst_tmp.append(lstArray[cc])
                    else:
                        lst_tmp.append(lstArray[cc][:,posY:])
                        print(lstArray[cc][:,posY:].shape)
                    cc += 1

                nArrayY =  np.concatenate(lst_tmp, axis= 1)
                print(f"#{cc} shape Y  {nArrayY.shape} ==> {np.max(nArrayY)}")

                lstGeral.append(nArrayY)

            else:
                lst_tmp = []
                for yy in range(0, quantY + 1):
                    # print((yy + 1) * dimPatch + 1)
                    if ((yy + 1)* dimPatch + 1) <  sideY:
                        lst_tmp.append(lstArray[cc][posX:, :])
                    else:
                        lst_tmp.append(lstArray[cc][posX:,posY:])
                    cc += 1
                nArrayY =  np.concatenate(lst_tmp, axis= 1)
                print(f"# {cc} shape Y  {nArrayY.shape} ==> {np.max(nArrayY)}")
                lstGeral.append(nArrayY)
        nArray =  np.concatenate(lstGeral, axis= 0)
        print("shape Array All  ", nArray.shape)               
    else:
        posInicX = int((sideX - dimPatch) /2 )
        posInicY = int((sideY - dimPatch) /2 )
        posEndcX = posInicX + dimPatch
        posEndcY = posInicY + dimPatch
        nArray = np.zeros((sideX, sideY))
        array_loaded = lstArray[0]
        print(f" array {lstnameArr[0]} readed {array_loaded.shape} ")
        nArray[posInicX: posEndcX, posInicY: posEndcY] = array_loaded
        print(f"know shape of array with  buffer > 0 --->  ", nArray.shape)
        


    save_array_to_TIF(patch_name, nArray, dictMetaD["transform"], myear)
    # return lstCoorArray



with open('regions_G2_panel_with_bufferv2.json', 'r') as json_file:
    dictFeatBuffer = json.load(json_file)

param = {
    'rutaInputImg': '/home/superuser/Dados/mapbiomas/mapping_areas_eolicas_fotovoltaicas/src/dados/patchs_pred_FV_g2C',
    # 'rutaInputImg': '/run/media/superuser/Almacen/mapbiomas/dadosCol9/fotoVol/fileTAR/patches_pred_FotoVv2/',
    # 'rutaInputImg': '/home/superuser/Dados/mapbiomas/dadosCol10/dbFotoV/patches_pred_FotoVv2',
    'rutaOutpatches': '/home/superuser/Dados/mapbiomas/mapping_areas_eolicas_fotovoltaicas/src/dados/array_pred_tif_g2C/',
    # 'rutaOutpatches': '/run/media/superuser/Almacen/mapbiomas/dadosCol9/fotoVol/patches_pred_FotoV/',#',   
    # 'rutaOutpatches': '/home/superuser/Dados/mapbiomas/dadosCol10/dbFotoV/predTIF_FotoVv2/',
    'rutaJson_atrib': '/home/superuser/Dados/mapbiomas/mapping_areas_eolicas_fotovoltaicas/src/preprocessing/metadataTIFv4_g2C',
    'transform': None,
    'projection': None,
    'metadata': None,
    'distM': 5,
    'quantAmostras' : 1000,
    "limPatches": 100,
    'size_patch': 256,
    'dirSaveBlock': True,
    'ordenBandas': ['B', 'G', 'R', 'N', 'iia', 'evi', 'ri', 'msavi', 'shape'],    
    'pInfo': False, # prints info of images 
    'exportPatchs': True
}

print("loaded files tif Alertas in: \n  ===> {}  ⌛...".format(
            param['rutaInputImg']))
fileslist = os.listdir(param['rutaInputImg'])
jsonlist = os.listdir(param['rutaJson_atrib'])


# path_MGRS = 'registrosDone.txt'
# lsPatchs = review_files_done(path_MGRS)

# arqFeitos = open(path_MGRS, 'a+')
lstFailsArr = []

for nyear in range(2015, 2026):
    for month in [6, 7, 9, 10, 11]:
        print(f"Processing Array of month {month} and Year {nyear} ")
        fileslistYY = []
        dict_imgName = {}
        #### ==========================================================================##
        #### separando todos os file dir correspondete ao ano nyear em uma lista unica ##
        print(f" show {len(fileslist)} size ")
        for nfile in fileslist:
            if str(nyear) in nfile and str(month) in nfile and nfile.endswith(".npy"):
                print("adding >>> ", nfile)
                fileslistYY.append(nfile)
        print("adding processs ", len(fileslistYY))
        # sys.exit()
        fileslistYY.sort()
        for cc, name_array in enumerate(fileslistYY):                   
            # print("-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+")
            # pos = name_array.find('g')
            # idCod = name_array[:pos]
            sufix = name_array.split("_")[-1]
            idKeys = name_array.replace("_" + sufix, "")
            idCod = name_array.split("_")[1]           
            lstKeyIds = list(dict_imgName.keys())
            
            if idKeys not in lstKeyIds:
                dict_imgName[idKeys] = [name_array]   
                print("Adding arrays to key >>>>> ", idKeys, " <<<<<<<<<")             
            else:
                lsttmp = list(dict_imgName[idKeys])
                lsttmp.append(name_array)
                # update dictionary in key idKeys
                dict_imgName[idKeys] = lsttmp
            
            print(f"... ☢ # {cc}: {name_array} >> {dictFeatBuffer[idCod]} ") # 

            # IniciarColeta(pathI, param, dictFeatBuffer[idCod])
            # arqFeitos.write(pathI + '\n')

            # sys.exit()
        # sys.exit()
        # Cria o diretório de destino se ele não existir
        pathDest = os.path.join(param['rutaOutpatches'], str(nyear))
        if not os.path.exists(pathDest):
            os.makedirs(pathDest)
    
        cc = 0
        for nkey, lstIds in dict_imgName.items():
            if cc > -1:
                print(f"# {cc} >>  joining > {nkey} >> \n  {lstIds}")
                # pos = name_array.find('g')
                idCod = nkey.split("_")[1]
                # idCod = nkey            
                print(f" >>>> {nkey}  >>> buffer {dictFeatBuffer[idCod]}")  # 
                # print(jsonlist[idCod + ".json"])
                try:
                    join_patchs(nkey, lstIds, dictFeatBuffer[idCod], param['size_patch'], nyear)  # 

                except:
                    print('erros iin ', nkey)
                    lstFailsArr.append(nkey)
            
            cc+= 1
        # sys.exit()
    print(f"=========== terminou year {nyear} ==========")
# print(list(jsonlist)[: 4])
for jj in lstFailsArr:
    print(jj)