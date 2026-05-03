import rasterio
# from os import system
import json
import os
import sys
import numpy as np
import time



def review_files_done(path_MGRSs):
    arqFeitos = open(path_MGRSs, 'r')
    lsPatches = []
    repetidor = 1
    textoImp = ''
    for ii in arqFeitos:
        ii = ii[:-1].replace(" ", "")
        if repetidor % 2 == 0:
            textoImp +=  ii 
            print (textoImp)
            textoImp = ''         
        else:        
            textoImp +=  ii + "|"         
        repetidor += 1
        lsPatches.append(ii)
    arqFeitos.close()
    return lsPatches

def expandAandYShapeArray( myArray, dimPatch):
    nArrayDim = np.zeros((dimPatch, dimPatch))
    nArrayDim[0: myArray.shape[0], 0: myArray.shape[1]] = myArray
    return nArrayDim

def expand_byOneSideArray( myArray, dimPatch, ladoX):
    if ladoX:
        nArrayDim = np.zeros((dimPatch, myArray.shape[1]))
        nArrayDim[:, 0: myArray.shape[1]] = myArray
    else:
        nArrayDim = np.zeros(( myArray.shape[0], dimPatch))
        nArrayDim[0: myArray.shape[0], :] = myArray
    
    return nArrayDim


def getlistPosofArraytoPatch(myArray, dimPatch):

    sideX = myArray.shape[0]
    sideY = myArray.shape[1]

    quantX = int(sideX / dimPatch)
    restoX = int(sideX % dimPatch)
    quantY = int(sideY / dimPatch)
    restoY = int(sideY % dimPatch)
    print(f" quantX {quantX} |  restoX  {restoX} ")
    print(f" quantY {quantY} |  restoY  {restoY} ")

    lstCoorArray = []
    if sideX > dimPatch or sideY > dimPatch:
        print("iterando por corte")
        for xx in range(0, quantX + 1):
            if ((xx + 1)* dimPatch + 1) <  sideX:
                for yy in range(0, quantY + 1):
                    # print((yy + 1) * dimPatch + 1)
                    if ((yy + 1)* dimPatch + 1) <  sideY:
                        coordArray = [xx * dimPatch, (xx + 1) * dimPatch, yy * dimPatch, (yy + 1) * dimPatch ]
                        # lstCoorArray.append(coordArray)
                        # print(f"yy > {yy}  | {coordArray}")
                    else:           
                        print("entrou ")
                        coordArray = [xx * dimPatch, (xx + 1) * dimPatch, myArray.shape[1] - dimPatch, myArray.shape[1]]
                        # print(f" adding ultimo {coordArray}")
                    lstCoorArray.append(coordArray)

            else:
                for yy in range(0, quantY + 1):
                    if ((yy + 1)* dimPatch + 1) <  sideY:
                        coordArray = [myArray.shape[0] - dimPatch, myArray.shape[0], yy * dimPatch, (yy + 1) * dimPatch ]
                    else:
                        coordArray = [myArray.shape[0] - dimPatch, myArray.shape[0], myArray.shape[1] - dimPatch, myArray.shape[1]]
                    lstCoorArray.append(coordArray)                 
    else:
        # esta imagem foi reconstruida por todos os lados        
        coordArray = [0, myArray.shape[0], 0, myArray.shape[1]]
        print("addicionado um unico valor de ", coordArray)
        lstCoorArray.append(coordArray)

    return lstCoorArray


def IniciarColeta(nameImg, pmtros, sizebuffer, yyear):
    ###############################################################
    ## help(https://rasterio.readthedocs.io)
    ## this allows rasterio to throw Python Exceptions
    ## Region "lectura Dados"
    ## listas de bandas ordenadas nas imagens               #######
    ## ['B', 'G', 'R', 'N', 'iia', 'evi', 'ri', 'gli', 'shape'] ###
    ## ----------------------------------------------------------##

    
    arrayImg = []
    listptos = []
    intX = 0
    intY = 0
    dictRegistros = {}

    path_ = pmtros['rutaInputImg']
    bandas = pmtros['ordenBandas']  # + ['class'] 
    
    try:
        # Open the dataset
        print(" reading in = ", path_ + "/" + nameImg)
        dataset = rasterio.open(path_ + "/" + nameImg)
        print(dataset.crs)
        dictRegistros['BoundingBox'] = dataset.bounds
        dictRegistros['transform'] = dataset.transform
        # dictRegistros['crs'] = dataset.crs
        # registrar o metadado da imagem em especifico 
        idCodShp = nameImg.split("_")[1]
        dictRegistros['idCod'] = idCodShp

        print("metadados da imagem ", pmtros['metadata'] )
        intY = dataset.width  # numero de rows 
        intX = dataset.height  # numero de cols 
        
        dictRegistros['xsize'] = intX
        dictRegistros['ysize'] = intY
        dictRegistros['buffer_size'] = sizebuffer

        print("Valor intX  horizontal columns / heigth", intX)
        print("valor intY vertical rows", intY)
        print(dictRegistros)
        nameJson = f"metadataTIFv4_g2C/{nameImg.replace('.tif', '')}.json"
        # verificando se o arquivo existe 
        if not os.path.isfile(nameJson):
            with open(nameJson, "w") as fp:
                json.dump(dictRegistros , fp) 
    
    except:
        print ('Unable to open INPUT.tif')
        # print (e)
        sys.exit(1)
    
    ##########################################################
    ## Se buffer > 0 significa que a imagem era menor que o ##
    ## tamanho do patchs, por tanto precissava aumentar o   ##
    ## de forma artificial, e a região coletada é maior que ##
    ## size patch, então centralizamos a região FotoV       ##
    ##########################################################
    if sizebuffer > 0:     
        # get patchs central            
        posInicX = int((intX - pmtros['size_patch']) /2 )
        posInicY = int((intY - pmtros['size_patch']) /2 )
        posEndcX = posInicX + pmtros['size_patch']
        posEndcY = posInicY + pmtros['size_patch']
        print("valores extremos carregados ")
        if (intX - 10) / pmtros['size_patch'] > 1 or (intY - 10) / pmtros['size_patch'] > 1: 
            sizebuffer = 0
    
    print("lindo o TIF " , dataset.indexes)
    
    for band_num  in range(1, len(bandas) + 1):         
        try:
            # print(" uff", sizebuffer)
            arr = dataset.read(band_num)          
            if sizebuffer > 0: 
                # print(" shape array ", arr.shape)
                ### get o patch no centro porque o resto foi aumentado artificialmente 
                print(f" {posInicX} : {posEndcX}, {posInicY} : {posEndcY}")
                arr = arr[posInicX: posEndcX, posInicY: posEndcY]         
            arrayImg.append(arr)     
            # print("carregou ok ", arr.shape)            

        except:
            # for example, try GetRasterBand(10)
            print ('Band {} not found'.format(band_num))
            print ()
            sys.exit(1)         


    # print("numero de imagens ", len(arrayImg))
    arrayImg = np.dstack(arrayImg)
    print("dimensions of bands array  ℛ³ : ", arrayImg.shape)
    # sys.exit()
    # reshape o array 
    # arrayImg = arrayImg.reshape(band.XSize, band.YSize, band_num)

    # print("dimensions of band array  ℛ² : ", arr.shape)
    

    if sizebuffer > 0: 
        print(" ===> update  ====")
        nameImgPatch = nameImg.replace('.tif', '') + '_g0'
        if pmtros['exportPatchs']:  
            save_datasetinNPZ(arrayImg, nameImgPatch, yyear)            

    else:
        # nodata = band.GetNoDataValue()
        lstCoordtoPatch = getlistPosofArraytoPatch(arrayImg, pmtros['size_patch'])
        print("lista de coordenadas ", lstCoordtoPatch)
        for cc, coords in enumerate(lstCoordtoPatch):
            X1 = coords[0]
            X2 = coords[1]
            Y1 = coords[2]
            Y2 = coords[3]
            arrayPatch = arrayImg[X1 : X2, Y1 : Y2, :]
            print("verificando o size Array builded ==> ", arrayPatch.shape)        
            nameImgPatch = nameImg.replace('.tif', '') + '_g' +  str(cc)
            # print(nameImgPatch)
            if pmtros['exportPatchs']:  
                save_datasetinNPZ(arrayPatch, nameImgPatch, yyear)
            print("    save ", nameImgPatch)
    # sys.exit()

    band = None
    dataset = None
    arrayImg = None
    arr = None
    
    print (" finish list ⚡⚡⚡")

def save_datasetinNPZ (arrayImg, nameMatriz, myear):
    
    if param["dirSaveBlock"]:
        # pathExp = os.path.join(param["rutaOutpatches"], str(myear))
        # verificar_pasta_exit(pathExp)
        pathExp = param["rutaOutpatches"]
        dirFile = os.path.join(pathExp, f"{nameMatriz}.npy")
        try:
            print("saving in >>> ", dirFile)
            np.save(dirFile, np.asarray(arrayImg, dtype= np.float32))
            # sys.exit()
        except :
            print("Matriz Patchs ", np.asarray(arrayImg).shape)
        
        del arrayImg
        # gc.collect()
        # sys.exit()

def verificar_pasta_exit(npath):
    # Verifique se a pasta existe
    # print(" >>>>>>>>>>>>>>> ", npath)
    if not os.path.exists(npath): # and not os.path.isdir(npath)
        print(f"A pasta '{npath}' não existe. Criando...")
        os.makedirs(npath)  # Cria a pasta e seus pais, se necessário
        print("Pasta criada com sucesso!")
    
    #     print(f"A pasta '{npath}' já existe.")
    # else:
    # sys.exit()

def verificar_numImagensYY(listPath):
    dictYY = {}
    for cc, pathI in enumerate(listPath[:]):        
        if pathI.endswith(".tif"):  #  and pathI not in lsPatchs
            nyear = pathI.split("_")[2].replace('.tif', '')
            keyList = list(dictYY.keys())
            if str(nyear) not in keyList:
                dictYY[str(nyear)] = 1
            else:
                dictYY[str(nyear)] += 1

    print("dictionario de year ")
    for nkey, nval in dictYY.items():
        print(f'Year >> {nkey} No. Imagens >> {nval}')



dictFeatBuffer = dict()
# Opening JSON file
with open('regions_G2_panel_with_bufferv2.json') as json_file:
    dictFeatBuffer = json.load(json_file)

param = {
    # 'rutaInputImg': '/run/media/superuser/Almacen/mapbiomas/dadosCol9/fotoVol/imgRegions_FotoV/imReg_FotoVv2y',
    # 'rutaOutpatches': '/run/media/superuser/Almacen/mapbiomas/dadosCol9/fotoVol/patches_FotoVv2y',#',  
    # 'rutaInputImg': '/home/superuser/Dados/mapbiomas/mapping_areas_eolicas_fotovoltaicas/src/dados/array_tif',
    # 'rutaOutpatches': '/home/superuser/Dados/mapbiomas/mapping_areas_eolicas_fotovoltaicas/src/dados/patchs_FV', 
    'rutaInputImg': '/home/superuser/Dados/mapbiomas/mapping_areas_eolicas_fotovoltaicas/src/dados/array_tif_g2C',
    'rutaOutpatches': '/home/superuser/Dados/mapbiomas/mapping_areas_eolicas_fotovoltaicas/src/dados/patchs_FV_g2C', 
    'transform': None,
    'projection': None,
    'metadata': None,
    'distM': 5,
    'quantAmostras' : 1000,
    "limPatches": 100,
    'size_patch': 256,
    'dirSaveBlock': True,
    'ordenBandas': ['B', 'G', 'R', 'N', 'iia', 'evi', 'ri', 'gli', 'shape'],    
    'pInfo': False, # prints info of images 
    'exportPatchs': True
}
# verify idCod with polygons regions 
# https://code.earthengine.google.com/7b6a3d6c320300088b535a7f192f6347


print("loaded files tif Alertas in: \n  ===> {}  ⌛...".format(
            param['rutaInputImg']))
fileslist = os.listdir(param['rutaInputImg'])
print(f" load files {len(list(fileslist))}")
lstBuffer0 = []
path_MGRS = 'registrosDone.txt'
lsPatchs = review_files_done(path_MGRS)
print(f"   >>> {len(lsPatchs)}")

verificar_numImagensYY(fileslist)
lstfilesArray = os.listdir(param['rutaOutpatches'])
lstArraysproc = []
for nameArr in lstfilesArray:
    parte = len(nameArr.split("_")[-1])
    if nameArr[: parte + 1] not in lstArraysproc:
        lstArraysproc.append(nameArr[: -(parte + 1)] + '.tif')

print(lstArraysproc[:2])
print(f"we have {len(lstArraysproc)} file array in folder ")
# sys.exit()
# nyear = '2024'
lstArraysproc = []
dd = 1
arqFeitos = open(path_MGRS, 'a+')
for nyear in range(2015, 2026):
    nyear = str(nyear)
    # if nyear != '2024':
    for cc, pathI in enumerate(fileslist[:]):
        print(pathI)
        # sys.exit()
        if pathI.endswith(".tif") and nyear in pathI:  #  and pathI not in lsPatchs
            try:
                print("-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+")
                # idCod = pathI.replace('.tif', '')
                idCod = pathI.split("_")[1]
                nyear = pathI.split("_")[2].replace('.tif', '')
                # if idCod in ['00000000000000000007', '00000000000000000008', '0000000000000000000a', '0000000000000000001f']:
                # if idCod in ['0000000000000000001b']:
                print(f"... ☢ # {cc}: >> {dd}  {idCod} year {nyear} >> buffer --> {dictFeatBuffer[idCod]}")
                # sys.exit()
                IniciarColeta(pathI, param, dictFeatBuffer[idCod], nyear)
                # setando tudo o mundo com buffer 0
                # if pathI not in lstArraysproc:
                #     IniciarColeta(pathI, param, 0, nyear)
                # arqFeitos.write(pathI + '\n')
                # if int(dictFeatBuffer[idCod]) == 0:
                #     IniciarColeta(pathI, param, dictFeatBuffer[idCod])
                    # lstBuffer0.append(idCod)
                time.sleep(1)
                dd += 1 
                sys.exit()
            except:
                print("Arquivo nao Salvo ")


    # if cc > 2:
    #     break
    # sys.exit()
    

for cc, iDbuff in enumerate(lstBuffer0):
    print(f"#{cc}  ====> {iDbuff} buffer 0")