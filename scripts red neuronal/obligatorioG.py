import pandas as pd
import numpy as np
import tensorflow as tf

from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from matplotlib import pyplot
from numpy import array
from sklearn.model_selection import train_test_split

from sklearn.metrics import *
from keras import backend
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Activation, BatchNormalization
from keras import regularizers
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import LeakyReLU
import random
from numpy.random import seed

from funcionesAux import *

indice_entrenamiento = 83
indice_test = 108
cant_pruebas = 1
#path = "C:\\Users\\usuario\\Desktop\\DatasetsPC\\Res_var_enT\\Gasoil\\"

path = "C:\\Users\\usuario\\Desktop\\ort\\obligatorio\\ResultadosRed2\\"

columna_objetivo ='consumo_miles_m3(t)'

#ordenados de menor a mayor
lrs = [0.0007]

batchs = [5]
epochss = [1]

arquitecturas = [([200, 150, 100], [50, 25]) ]

experimentos = crearExperimentos(lrs, batchs, epochss, arquitecturas)

cant_meses = indice_test - indice_entrenamiento

tf.random.set_seed(40)
random.seed(40)
np.random.seed(40)

series = pd.read_csv("C:\\Users\\usuario\\Desktop\\ort\\obligatorio\\dataset\\gasoil_csv.csv",sep = ',',header = 0)
meses = pd.read_csv("C:\\Users\\usuario\\Desktop\\ort\\obligatorio\\dataset\\meses.csv",sep = ',',header = 0)
diasVentasRC = pd.read_csv("C:\\Users\\usuario\\Desktop\\ort\\obligatorio\\dataset\\dias_venta_teoricos.csv",sep =',',header = 0)

#ESCALAMIENTO DE MESES
scaler = MinMaxScaler(feature_range = (0, 1))
df_scaled = pd.DataFrame(scaler.fit_transform(meses), columns = meses.columns)
df_scaled

#ESCALAMIENTO DE DIAS DE VENTAS reales
scaler = MinMaxScaler(feature_range = (0, 1))
df_scaled2 = pd.DataFrame(scaler.fit_transform(diasVentasRC), columns = diasVentasRC.columns)
df_scaled2

#UNION DE DATAFRAME "series" y "meses escalado"
seriesF=pd.concat([series, df_scaled], axis = 1)
seriesF=pd.concat([seriesF, df_scaled2], axis = 1)
seriesF

n=24
series=series_to_supervised(seriesF,n,1)
series.head()

series=series.drop('Fecha(t-%d)' % 1, axis=1)
for i in range(2,n+1):
    aux='Fecha(t-%d)' % i
    series = series.drop(aux, axis=1)
series.head()


aux1 ='Dias de Venta(t)' 
series = series.drop(aux1, axis=1)

aux2 ='Meses(t)' 
series = series.drop(aux2, axis=1)

series = series.reset_index()
series

inputcols = ["Dias de Venta(t-24)","consumo_miles_m3(t-24)","Meses(t-24)",
"Dias de Venta(t-23)","consumo_miles_m3(t-23)","Meses(t-23)","Dias de Venta(t-22)",
"consumo_miles_m3(t-22)","Meses(t-22)","Dias de Venta(t-21)","consumo_miles_m3(t-21)","Meses(t-21)",
"Dias de Venta(t-20)","consumo_miles_m3(t-20)","Meses(t-20)","Dias de Venta(t-19)",
"consumo_miles_m3(t-19)","Meses(t-19)","Dias de Venta(t-18)","consumo_miles_m3(t-18)","Meses(t-18)",
"Dias de Venta(t-17)","consumo_miles_m3(t-17)","Meses(t-17)","Dias de Venta(t-16)",
"consumo_miles_m3(t-16)","Meses(t-16)","Dias de Venta(t-15)","consumo_miles_m3(t-15)",
"Meses(t-15)","Dias de Venta(t-14)","consumo_miles_m3(t-14)","Meses(t-14)","Dias de Venta(t-13)",
"consumo_miles_m3(t-13)","Meses(t-13)","Dias de Venta(t-12)","consumo_miles_m3(t-12)",
"Meses(t-12)","Dias de Venta(t-11)","consumo_miles_m3(t-11)","Meses(t-11)","Dias de Venta(t-10)",
"consumo_miles_m3(t-10)","Meses(t-10)","Dias de Venta(t-9)","consumo_miles_m3(t-9)","Meses(t-9)",
"Dias de Venta(t-8)","consumo_miles_m3(t-8)","Meses(t-8)","Dias de Venta(t-7)","consumo_miles_m3(t-7)",
"Meses(t-7)","Dias de Venta(t-6)","consumo_miles_m3(t-6)","Meses(t-6)", "Dias de Venta(t-5)",
"consumo_miles_m3(t-5)","Meses(t-5)", "Dias de Venta(t-4)","consumo_miles_m3(t-4)","Meses(t-4)", 
"Dias de Venta(t-3)","consumo_miles_m3(t-3)","Meses(t-3)","Dias de Venta(t-2)","consumo_miles_m3(t-2)",
"Meses(t-2)","Dias de Venta(t-1)","consumo_miles_m3(t-1)","Meses(t-1)"]


etiqueta =["consumo_miles_m3(t)"]

train = series.loc[:indice_entrenamiento]
test = series.loc[indice_entrenamiento+1:indice_test]


x_train = train[inputcols].values
y_train = train[etiqueta].values

x_test = test[inputcols].values
y_test = test[etiqueta].values

x_train = np.array(x_train)
x_train = np.reshape(x_train, (x_train.shape[0], n, int(x_train.shape[1]/n)))

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], n, int(x_test.shape[1]/n)))


callbacks = EarlyStopping(monitor='rmse', patience=5)

for e in experimentos:
    batch = e["batch"]
    lr=e["learning_rate"]
    epochs=e["epochs"]
    lstms=e["arquitectura"][0]
    densas=e["arquitectura"][1]

    input_s = (x_train.shape[1], x_train.shape[2])
    listAuxGasoil=[]
   
    predGasoil=[]
    mgasoil=[]
    
    for i in range(cant_pruebas):
        
        scoresMarianaGasoil = []

        modeloGasoil = crearModelo(input_s,lstms,densas,1,lr,True,0.2,0.001)      
        modeloGasoil.fit(x_train, y_train,epochs = epochs, batch_size = batch) 
        
        scoresGasoil = modeloGasoil.evaluate(x_test,y_test)
        predictionsGasoil = modeloGasoil.predict(x_test)

        scoresMarianaGasoil.append(scoresGasoil[1])

        predGasoil.append(predictionsGasoil)
        emensuales=erroresMensuales(cant_meses,predictionsGasoil,test[columna_objetivo])
        mgasoil.append(emensuales)
        listAuxGasoil.append(np.mean(emensuales))

    generacionArchivo(e, path,listAuxGasoil,cant_meses,predGasoil,mgasoil)






        
