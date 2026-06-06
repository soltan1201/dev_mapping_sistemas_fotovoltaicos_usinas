#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
import sys
import glob
import json
import time
import numpy as np
import argparse
# Tensorflow setup.
import tensorflow as tf
from pathlib import Path
print("We working with TF Version = ", tf.__version__)
# tf.enable_eager_execution()

from tensorflow import keras
from keras import regularizers
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose
from keras.layers import BatchNormalization, Dropout, Lambda, UpSampling2D
from keras.optimizers import Adam
from keras.layers import Activation, MaxPool2D, Concatenate

# FEATURE_BANDS   = ['blue', 'green', 'red', 'nir', 'pvi', 'iia', 'ri', 'evi']
indexBands   = ['blue', 'green', 'red', 'nir', 'pvi', 'iia', 'ri', 'evi']

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

class make_array_Predictions(object):
    path_default = ''
    pathOutput = 'array_predictV3'
    XSize = 256
    YSize = 256

    def __init__(self, model_path, path_patchsOut, path_Base):
        self.pathOutput = path_patchsOut
        self.path_Base = path_Base
        print(" patch output label predict ", self.pathOutput)
        # Recreate the exact same model, including its weights and the optimizer
        # Métricas customizadas (incluindo Jaccard Index)
        def jaccard_index(y_true, y_pred):
            y_pred = tf.cast(y_pred > 0.5, tf.float32)  # Binariza as predições
            intersection = tf.reduce_sum(y_true * y_pred)
            union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
            return intersection / (union + tf.keras.backend.epsilon())

        def dice_loss(y_true, y_pred):
            smooth = 1e-5
            y_pred = tf.math.sigmoid(y_pred)  # Remove se já houver sigmoid na saída
            intersection = tf.reduce_sum(y_true * y_pred)
            return 1 - (2.0 * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

        # Passo 2: Registre a função manualmente (se não usou o decorador antes de salvar)
        keras.utils.get_custom_objects()["dice_loss"] = dice_loss
        keras.utils.get_custom_objects()["jaccard_index"] = jaccard_index
        try:
            self.mUNETtrained = tf.keras.models.load_model(
                                    model_path, 
                                    custom_objects={
                                        'jaccard_index': jaccard_index,  # Função definida anteriormente
                                        'loss': dice_loss  # Se houver funções customizadas na loss
                                    })
            print(" ---- model loaded UNet treinado carregado ----")            
        except Exception as e:
            print("Erro:", e)
        # Show the model architecture
        # self.mUNETtrained = self.get_model()
        # self.mUNETtrained = self.mUNETtrained.lo
        # print(self.mUNETtrained.summary())

    def save_datasetinNPY (self, arrayImg, nameMatriz):
        dirFile = os.path.join(self.pathOutput, nameMatriz)
        # print("saving in >> ", dirFile)
        # print( self.pathOutput)
        try:
            np.save(dirFile, np.asarray(arrayImg, dtype= np.uint8))
            # print("saved")
        except :
            print("fail Matriz Patchs ", np.asarray(arrayImg).shape)
        del arrayImg

    def load_and_predict_arrayPatchs(self, path_matrix):
        arrFails = True
        # the array is saved in the file geekfile.npy
        name_matrix = path_matrix.split("/")[-1]
        try:
            print(path_matrix)
            array_pred = np.load(path_matrix)
            print("shape matriz loaded ", array_pred.shape)
            array_pred = np.array(array_pred[:, :, :9]).reshape(1,256,256,9)
            array_pred = array_pred / 10000
            # print("shape matrix ", array_pred.shape)
            predict_raster = self.mUNETtrained.predict(array_pred, steps=1, verbose=2)
            # print("shape Predict ", predict_raster.shape)
            predict_raster = predict_raster[0,:,:,0]
            # print(predict_raster.shape)
            predict_raster[predict_raster < 0.6] = 0
            predict_raster[predict_raster >= 0.6] = 1
            predict_raster = predict_raster.astype(np.uint8)
            # predict_raster = predict_raster * 255
            print(f" {type(predict_raster)}  shape the image predict ", predict_raster.shape, "  maximo =  ", np.max(predict_raster) )
            self.save_datasetinNPY(predict_raster, name_matrix)
        except:
            print(" array fail ", name_matrix)
            arrFails = False

        return arrFails



    
   
    


# path_patchInput = '/home/superuser/Dados/mapbiomas/dadosCol10/dbFotoV/patches_FotoVv2y'
# path_patchOutput = '/home/superuser/Dados/mapbiomas/dadosCol10/dbFotoV/patchs_pred_FV3'
path_patchInput = '/home/superuser/Dados/mapbiomas/mapping_areas_eolicas_fotovoltaicas/src/dados/patchs_FV_g2'
path_patchOutput = '/home/superuser/Dados/mapbiomas/mapping_areas_eolicas_fotovoltaicas/src/dados/patchs_pred_FV3_g2'

parser = argparse.ArgumentParser()
parser.add_argument('path_input_output', type=str,  default=(path_patchInput, path_patchOutput), help="digite  os caminhos de repositor e guardado das imagenes patchs: path_input, path_output " )
try:
    args = parser.parse_args()
    listPaths = args.path_input_output
    listPaths= listPaths.split(',')
except argparse.ArgumentTypeError as e:
    print(f"Invalid argument: {e}")

    # if not os.path.exists(npath)
if os.path.exists(listPaths[0]) and os.path.exists(listPaths[1]):
    path_patchInput = listPaths[0]
    path_patchOutput  = listPaths[1]
    print(" Path folder of input and output are loaded ")
else:
    print("Por reload the script with correct parameters ")
    sys.exit()

pathbase = str(Path(os.getcwd()).parents[0])
print("path base ", pathbase)
lstfilepath = glob.glob(path_patchInput + '/*.npy')
print(f"we load {len(lstfilepath)} file npy")
print("showing the 5 first ")
for ii, mpath in enumerate(lstfilepath[:5]):
    print(f" #{ii}  >> {mpath}")
# # Métricas customizadas (incluindo Jaccard Index)
# def jaccard_index(y_true, y_pred):
#     y_pred = tf.cast(y_pred > 0.5, tf.float32)  # Binariza as predições
#     intersection = tf.reduce_sum(y_true * y_pred)
#     union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
#     return intersection / (union + tf.keras.backend.epsilon())

# def dice_loss(y_true, y_pred):
#     smooth = 1e-5
#     y_pred = tf.math.sigmoid(y_pred)  # Remove se já houver sigmoid na saída
#     intersection = tf.reduce_sum(y_true * y_pred)
#     return 1 - (2.0 * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


keras_MODEL_DIR = os.path.join(pathbase, 'models/model_accuracy_25B_04_2025.keras')
print("reading model from ", keras_MODEL_DIR)
# sys.exit()
makepredictionclass = make_array_Predictions(keras_MODEL_DIR, path_patchOutput, path_patchInput)
time.sleep(5)


lstArrayFails = []
dictRegistros = {}
# nyear = '2023'
for nyear in range(2024,2023,-1):
    nyear = str(nyear)
    nlstfilepath = [npath for npath in lstfilepath if nyear in npath]
    nTotal = len(nlstfilepath)
    print(" total de files ", nTotal)
    # sys.exit()
    for cc, namefile in enumerate(nlstfilepath[:]):
        name_array = namefile.replace(path_patchInput + '/', '')
        print(f"# {cc}/{nTotal} processando = {name_array}")
        idCod = name_array.split("_")[1] 
        if idCod in ['0000000000000000001b']:
            array_process =  makepredictionclass.load_and_predict_arrayPatchs(namefile)
            if not array_process :
                lstArrayFails.append(name_array)
                dictRegistros[name_array] = 'fails'
                print(f" {name_array} >> falloooo <<< ")
            else:
                dictRegistros[name_array] = 'saved'

# Opening JSON file
# with open(os.path.join(pathbase, 'registro_patchs_predicted_g2.json'), 'w') as json_file:
#     json.dump(dictRegistros , json_file) 