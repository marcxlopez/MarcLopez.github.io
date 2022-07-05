# -*- coding: utf-8 -*-
"""
Created on Tue May 17 18:24:11 2022

@author: marcl
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

## Librerias para los clusters
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score,accuracy_score
from sklearn.cluster import KMeans

from sklearn.tree import _tree, DecisionTreeClassifier


#-----------------------------------------------------------------------
#Load the data and look at the first few rows
PATH = "C:\\Users\marcl\\Desktop\\TFG\\GITHUB TFG\\"
DATASETS_DIR = PATH + "data\\"

# =============================================================================
# Cargamos la base de datos
hotelesNorm = pd.read_pickle(DATASETS_DIR + 'HotelesNormalizados.pkl')
hoteles = pd.read_pickle(DATASETS_DIR + 'HotelesImputados.pkl')

###Bucle para realizar diferentes clusterings KMEANS 
#comparar COEFICIENTE DE SILHOUETTE

for k in range(2,8):
    #realizar el clustering
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(hotelesNorm)
    #calcular el silhouette score
    silhouette_avg = silhouette_score(hotelesNorm, kmeans.labels_)
    print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)

#------------------------------------------------------------------------------
##Realizamos KMEANS con 2 clusters 
#porque el COEFICIENTE DE SILHOUETTE es demasiado similar entre grupos 
kmeans = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 300,
                n_init = 10, random_state = 0)
kmeans.fit(hotelesNorm)
labels_pred = kmeans.labels_
labels_true = hoteles['precios']

# Graficamos los valores de la mejor clasificación
plt.figure(figsize = (10, 7))
plt.scatter(hotelesNorm['precios'], hotelesNorm['distancia'],
            c = kmeans.labels_) 
plt.xlabel('precios')
plt.ylabel('distancia')
plt.title('Clustering K-Means con k = 2')
plt.legend(range(1, kOptima + 1))

###Adjusted Rand Index 
labels_pred = kmeans.labels_
labels_true = hoteles['precios']
from sklearn.metrics.cluster import adjusted_rand_score
adjusted_rand_score(labels_true, labels_pred)
#------------------------------------------------------------------------------
#Realizamos KMEANS con 3 clusters
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300,
                n_init = 10, random_state = 0)
kmeans.fit(hotelesNorm)
labels_pred = kmeans.labels_
labels_true = hoteles['precios']

# Graficamos los valores de la mejor clasificación
plt.figure(figsize = (10, 7))
plt.scatter(hotelesNorm['precios'], hotelesNorm['distancia'],
            c = kmeans.labels_) 
plt.xlabel('precios')
plt.ylabel('distancia')
plt.title('Clustering K-Means con k = 3')
plt.legend(range(1, kOptima + 1))
###Adjusted Rand Index 
labels_pred = kmeans.labels_
labels_true = hoteles['precios']
from sklearn.metrics.cluster import adjusted_rand_score
adjusted_rand_score(labels_true, labels_pred)
#------------------------------------------------------------------------------
#Realizamos KMEANS para 6 clusters 
kmeans = KMeans(n_clusters = 6, init = 'k-means++', max_iter = 300,
                n_init = 10, random_state = 0)
kmeans.fit(hotelesNorm)
labels_pred = kmeans.labels_
labels_true = hoteles['precios']

# Graficamos los valores de la mejor clasificación
plt.figure(figsize = (10, 7))
plt.scatter(hotelesNorm['precios'], hotelesNorm['distancia'],
            c = kmeans.labels_) 
plt.xlabel('precios')
plt.ylabel('distancia')
plt.title('Clustering K-Means con k = 6')
plt.legend(range(1, kOptima + 1))
###Adjusted Rand Index 
labels_pred = kmeans.labels_
labels_true = hoteles['precios']
from sklearn.metrics.cluster import adjusted_rand_score
adjusted_rand_score(labels_true, labels_pred)