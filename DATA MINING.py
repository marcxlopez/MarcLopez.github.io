# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 17:55:08 2022

@author: marcl
"""

# =============================================================================
# Cargamos las librerias necesarias
import pandas as pd
import re
import string
from tabulate import tabulate
import geopy.distance
from datetime import timedelta

# Cargamos las librerias necesarias
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

## Librerias necesarias para la imputación de valores faltantes
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression

## Librerias para la normalización de los datos
from sklearn.preprocessing import normalize

# PARAMETROS:
PATH = "C:\\Users\marcl\\Desktop\\TFG\\GITHUB TFG\\"
DATASETS_DIR = PATH + "data\\"
OUTPUT_DIR = PATH + "output\\"

# =============================================================================
# Cargamos los datos que hemos scrappeado
hoteles1 = pd.read_pickle(DATASETS_DIR + 'HotelesDATA.pkl')
hoteles = hoteles1.reset_index()
# =============================================================================
# Realizamos el PREPROCESSING:
### Hotel 
hoteles = hoteles.dropna(axis = 0, subset = ['coordenadas'], )
hotelPR = pd.DataFrame(hoteles.Hotel)

# -----------------------------------------------------------------------------
### CheckIn
# nos aseguramos de que omisiones no generen error 
fechaIn = pd.to_datetime(hoteles.checkIn, format='%Y-%m-%d', errors = 'coerce')
fechaOut = pd.to_datetime(hoteles.checkOut, format='%Y-%m-%d', errors = 'coerce')

day_cin = fechaIn.dt.weekday
month_cin = fechaIn.dt.month

#creamos las dos variables de check out 

#### Calcular dia de la semana
hotelPR['MesEntrada'] = [month_cin[i] for i in month_cin]

#### Calcular el mes
hotelPR['DiaEntrada'] = [day_cin[x] for x in day_cin]

# -----------------------------------------------------------------------------

### CheckOut
day_cout = fechaOut.dt.weekday
month_cout = fechaOut.dt.month

#### Calcular dia de la semana
hotelPR['DiaSalida'] = [day_cout[i] for i in day_cout]
hotelPR['MesSalida'] = [month_cout[x] for x in month_cout]


# -----------------------------------------------------------------------------
### Estrellas
estrellas = [re.sub(" estrellas| estrella", "", estr) for estr in hoteles['Estrellas']]
hotelPR['Estrellas'] = [float(estr) if estr != '' else float("nan") for estr in estrellas ]

# -----------------------------------------------------------------------------
### Ratio 
ratio = [re.sub("\n.", "", rat) for rat in hoteles['Ratio']]
hotelPR['Ratio'] = [float(re.sub(",", "", rat)) if rat != '' else float("nan") for rat in ratio]

# -----------------------------------------------------------------------------
### Ratio_descr 
#print(tabulate(pd.crosstab(index = hoteles['Ratio_descr'], columns = "count"), 
               #headers = 'firstrow', tablefmt = 'fancy_grid'))
hotelPR['ratioDescr'] = hoteles['Ratio_descr']

# -----------------------------------------------------------------------------
### Ammenities #####
ammenities = [am.split("\n") for am in hoteles['Ammenities']]
df = pd.get_dummies(pd.DataFrame(ammenities))
df.columns = df.columns.str.split("_").str[-1]
df = df.groupby(df.columns.map(string.capwords), axis=1).sum()
hotelPR = pd.concat([hoteles, df], axis = 1) ### y los NaN porque vuelven??

# -----------------------------------------------------------------------------
### lugaresInteres 
lugaresInteres = [am.split("\n") for am in hoteles['lugaresInteres']]
lugaresInteres = [am.split(":")[0] for ams in lugaresInteres for am in ams]
lugaresInteres = list(set(lugaresInteres))

# -----------------------------------------------------------------------------
### tamaño 
#### plantas
plantas = [re.findall(r'[0-9]+ planta', x) for x in hoteles['tamanyo']]
plantas = [re.sub(" planta|\[|\]|\'", "", str(p)) for p in plantas]
hotelPR['Plantas'] = [int(plt) if plt != '' else float("nan") for plt in plantas]

habitaciones = [re.findall(r'[0-9]+ habita', x) for x in  hoteles['tamanyo']]
habitaciones = [re.sub(" habita|\[|\]|\'", "", str(h)) for h in habitaciones]
hotelPR['habitaciones'] = [int(hab) if hab != '' else float("nan") for hab in habitaciones]

hotelPR['ratioHabPlanta'] = hotelPR['habitaciones']/hotelPR['Plantas']
        
# -----------------------------------------------------------------------------
### coordenadas 

hotelPR['latitud'] = [float(hot[0].split(",")[0]) for hot in hoteles['coordenadas']] 
hotelPR['longitud'] = [float(hot[0].split(",")[1]) for hot in hoteles['coordenadas']]

#### Calculamos la distancia entre puntos de interes
#creamos DataFrame con los lugares de interes
nombre = ['Dalt Vila','Ses Salines',
'Sant Antoni de Portmany',"Cala d'Hort",'Puerto de Ibiza',
'Santa Eulária des Riu','Cala Benirrás','Sant Joan de Labritja','Cala Portinatx',
'Sant Josep de sa Talaia','Aeropuerto de Ibiza','Cala Comte','Cala de Sant Vicent']
latitud = [38.906613,38.5030,38.9806800,38.890,38.9089,38.9833,39.107,39.0778,39.1104253,38.9217,38.87294785,38.9634,39.0433]
longitud = [1.436293,1.2318,1.3036200,1.2235,1.43238,1.51667,1.5374,1.51227,1.5181252, 1.29344,1.370372,1.2246, 1.3524]

#crear dataframe lugares_interes con los datos de nombre, latitud y longitud
lugares_interes = pd.DataFrame({'nombre':nombre,'latitud':latitud,'longitud':longitud})

## Añadir estos valores en el data frame lugares_interes
lugares_interes = lugares_interes.append({'nombre': 'Ayuntamiento', 'latitud': 38.9070794, 'longitud': 1.4292239}, ignore_index=True)

for j in range(0, lugares_interes.shape[0]): #realizar bucle tantas veces como lugares de interes haya
    distancia = []    
    # Realizamos el bucle para todos los hoteles de la base de datos
    coordComparar = (lugares_interes.latitud[j], lugares_interes.longitud[j])
    for i in range(0, hotelPR.shape[0]): #realizar bucle tantas veces como hoteles haya 
        coords_2 = (hotelPR.latitud[i], hotelPR.longitud[i])
        distancia.append(geopy.distance.geodesic(coordComparar, coords_2).km)

    # Añadimos las distancias calculadas al dataframe de distancias ( me falta saber que qu)
    hotelPR['Prox_' + lugares_interes.nombre[j]] = distancia
    
# -----------------------------------------------------------------------------
###HOTEL PRECIO (funciona, pero hay bastantes NaN)
#extraer los valores anteriores al simbolo del euro
hotelPR['precios'] = [re.sub("\€", "", x) for x in hoteles['precio']]
#si hotelPR['precios'] tiene un simbolo de %, extraemos el valor 
hotelPR['precios'] = [re.sub("\%", "", x) for x in hotelPR['precios']]
#convertimos a float hotelPR['precios']
hotelPR['precios'] = [float(x) if x != '' else None for x in hotelPR['precios']]
#eliminamos los valores negativos 
hotelPR['precios'] = [x if x > 0 else float("nan") for x in hotelPR['precios']]

# =============================================================================
# Guardamos en formato de pickle
hotelPR.to_pickle(DATASETS_DIR + "HotelesPreprocesados.pkl")
hotelPR.to_csv(DATASETS_DIR + "HotelesPreprocesados.csv",";")

# =============================================================================
### N1. NORMALIZAR DATOS 
# crear matriz numerica 
datan = hotelesDist.drop(['Hotel','ratioDescr'], axis = 1) 
# Base de datos que debemos de imputar
datanm = datan[['Plantas','habitaciones']]

#-----------------------------------------------------------------------
# Creamos el estimador lineal
lr = LinearRegression()
imp = IterativeImputer(estimator = lr, missing_values = np.nan, 
                       max_iter = 30, verbose = 2, 
                       imputation_order = 'roman', random_state = 0)
X = imp.fit_transform(datanm)
X[:, 0] = np.around(X[:, 0], decimals = 0)
X[:, 1] = np.round(X[:, 1], decimals = 0)

#-----------------------------------------------------------------------
# introducir X en datan como 'Plantas', 'habitaciones', 'ratioHabPlanta'
datan['Plantas'] = X[:, 0]
datan['habitaciones'] = X[:, 1]
datan['ratioHabPlanta'] = datan['habitaciones']/datan['Plantas']

#quitar los datos omitidos (RATIO)
#quitar nan en datan
datan = datan.dropna()

# =============================================================================
# Añadimos las dos variables faltantes a nuestro datan
datan['Hotel'] = hoteles['Hotel']
datan['ratioDescr'] = hoteles['ratioDescr']

# =============================================================================
# Guardamos en formato de pickle la base de datos 
datan.to_pickle(DATASETS_DIR + "HotelesImputados.pkl")
datan.to_csv(DATASETS_DIR + "HotelesImputados.csv",";")

# =============================================================================
## NORMALIZAR DATOS 
data_scaled = datan.drop(['Hotel','ratioDescr'], axis = 1) 
columnas = data_scaled.columns
data_scaled = normalize(data_scaled)
data_scaled = pd.DataFrame(data_scaled, columns = columnas)
data_scaled.head()

## Calculamos la correlación de los datos
corr = data_scaled.corr()

# =============================================================================
# Guardamos la base de datos normalizada
data_scaled.to_pickle(DATASETS_DIR + "HotelesNormalizados.pkl")
data_scaled.to_csv(DATASETS_DIR + "HotelesNormalizados.csv", ";")

