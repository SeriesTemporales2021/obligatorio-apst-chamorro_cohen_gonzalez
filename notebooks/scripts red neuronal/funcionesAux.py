import numpy as np
from keras import backend
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras import regularizers
from keras.layers import *
from keras.models import Sequential
import pandas as pd
from pandas import *


def series_to_supervised(data, window = 1, lag = 1, dropnan = True):
    data=DataFrame(data)
    cols, names = list(), list()
    # Input sequence (t-n, ... t-1)
    for i in range(window, 0, -1):
        cols.append(data.shift(i))
        names += [('%s(t-%d)' % (col, i)) for col in data.columns]
        
    for i in range(0,lag):
        cols.append(data.shift(-i))
        if i == 0:
            names += [('%s(t)' % (col)) for col in data.columns]
        else:
            names += [('%s(t+%d)' % (col, i)) for col in data.columns]

    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # Drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def generar_inputcols(producto, n = 12):
    inputcols = []
    for i in reversed(range(1, n + 1)):
        inputcols.append("Dias de Venta(t-{})".format(i))
        inputcols.append("{}(t-{})".format(producto, i))
        inputcols.append("Meses(t-{})".format(i))
    return inputcols


def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

def mariana(y_true, y_pred):
    return backend.mean(backend.abs(y_pred/y_true-1)*100, axis=-1)


def crearModelo(input_s,neuronas_LSTM=[],neuronas_D=[],cant_salida=1,lr=1e-04,bn=True,do=0.2,kernel_reg=0.01):

    opt = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)

    modelo = Sequential()

    if bn:
        modelo.add(BatchNormalization(input_shape = input_s))
    if len(neuronas_LSTM) > 1:
        modelo.add(LSTM(units = neuronas_LSTM[0], activation= LeakyReLU(alpha=0.1), return_sequences = True,
            kernel_regularizer = regularizers.l1(kernel_reg),input_shape = input_s))
    elif len(neuronas_LSTM) > 0:
        modelo.add(LSTM(units = neuronas_LSTM[0], activation= LeakyReLU(alpha=0.1), return_sequences = False,
            kernel_regularizer = regularizers.l1(kernel_reg),input_shape = input_s))
    elif len(neuronas_D) > 0:
        modelo.add(Dense(units = neuronas_D[0],kernel_regularizer=regularizers.l1(kernel_reg),activation=LeakyReLU(alpha=0.1),input_shape=input_s))

    modelo.add(Dropout(do))

    for lstm in neuronas_LSTM[1:len(neuronas_LSTM)-1]:
        modelo.add(LSTM(units = lstm, return_sequences = True, activation= LeakyReLU(alpha=0.1)))

    if len(neuronas_LSTM) > 1:
        modelo.add(LSTM(units = neuronas_LSTM[-1], return_sequences = False, activation= LeakyReLU(alpha=0.1)))

    ini=0
    if len(neuronas_LSTM) == 0:
        ini=1

    for dens in neuronas_D[ini:]:
        modelo.add(Dense(units = dens,activation=LeakyReLU(alpha=0.1)))

    modelo.add(Dense(units = cant_salida,activation='linear'))

    modelo.compile(loss = mariana,optimizer = opt , metrics=[mariana])

    return modelo


def erroresMensuales(cant_meses,predicciones,vreales):
    dfF=pd.DataFrame()
    dfF['Valores Reales']=vreales
    dfF['Predicciones']=predicciones

    errores=[]
    for i in range(cant_meses):
        error=abs(dfF.iloc[i]['Predicciones']/dfF.iloc[i]['Valores Reales']-1)*100
        errores.append(error)

    return errores

def crearExperimentos(lrs,batchs,epochss,arquitecturas):
    experimentos = []
    for lr in lrs:
        for batch in batchs:
            for epochs in epochss:
                for arquitectura in arquitecturas:
                    if (len(arquitectura[0]) > 0) and (len(arquitectura[1]) > 0):
                        nombre = "lstms_densas_{}".format(batch)
                    elif (len(arquitectura[0]) > 0):
                        nombre = "lstms_{}".format(batch)
                    elif len(arquitectura[1]) > 0:
                        nombre = "densas_{}".format(batch)
                    else:
                        nombre = "1_densa_{}".format(batch)
                    e = {
                            "nombre":nombre,
                            "learning_rate": lr,
                            "batch": batch,
                            "epochs": epochs,
                            "arquitectura":arquitectura
                        }
                    experimentos.append(e)
    return experimentos

def generacionArchivo(e, path,listaAux,cant_meses,preds,errores):   

    diccionarioSalida={"ept":np.mean(listaAux),
                        "epa":listaAux,
                        "em":errores
                       }

    archivoSalida = open ("{}{}_{}_{}_{}.txt".\
                    format(path, e["nombre"], \
                        round(np.mean(listaAux),3), \
                        round(np.min(listaAux),3), \
                        round(np.max(listaAux),3)), "w")

    archivoSalida.write("{}\n\n".format(diccionarioSalida))

    archivoSalida.write("Arquitectura:\nLSTM: {}\nDensas: {}\n\nEpochs: {}\nLearning Rate: {}\nBatch: {}\n\n". \
                  format(e["arquitectura"][0],e["arquitectura"][1],e["epochs"],e["learning_rate"],e["batch"]))

    archivoSalida.write("Error Promedio Total: {}\n\n".format(np.mean(listaAux)))

    i=0
    for promedio in listaAux:
        i+=1
        archivoSalida.write("Error promedio anual ejecucion {}: {}\n".format(i,promedio))
    i=0
    for promedio in listaAux:
        i+=1
        archivoSalida.write("\nError promedio anual ejecucion {}: {}\n".format(i,promedio))
        for j in range(cant_meses):
            mes={0:"Enero",1:"Febrero",2:"Marzo", 3:"Abril", 4:"Mayo", 5:"Junio", 
                 6:"Julio", 7:"Agosto", 8:"Septiembre",9:"Octubre", 10:"Noviembre", 11:"Diciembre"}[j]
            archivoSalida.write("Mes {}: {} - {}\n".format(mes, preds[i-1][j][0], errores[i-1][j]))

    archivoSalida.close()