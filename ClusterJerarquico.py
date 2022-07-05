# -*- coding: utf-8 -*-
"""
Created on Wed May 11 16:50:50 2022

@author: Marc Lopez
"""

# Cargamos las librerias necesarias
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

# =============================================================================
#Let’s first draw the dendrogram to help us 
#decide the number of clusters for this particular problem

plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
dend = shc.dendrogram(shc.linkage(hotelesNorm, method='ward'))

silluete = []
kden = range(2, 7)

for k in kden:
    print("Realizamos agrupación de k = " + str(k))
    
    # Calculamos la clasificación con el número k 
    cluster = AgglomerativeClustering(n_clusters = k, affinity = 'euclidean', 
                                      linkage = 'ward')  
    # Predecimos el número de clases
    cluster.fit_predict(hotelesNorm)

    # Gráficamos el corte del dendograma
    plt.figure(figsize=(10, 7))  
    plt.title("Dendograma para k = " + str(k))  
    dend = shc.dendrogram(shc.linkage(hotelesNorm, method = 'ward', 
                                      metric = 'euclidean'), labels = cluster.labels_)
    plt.axhline(y = 6, color = 'r', linestyle = '--')

    # Graficamos dicho corte en las variables precio vs. distnacia en la variable original
    # plt.figure(figsize = (10, 7))  
    # plt.scatter(hotelesNorm['precio'], hotelesNorm['distancia'], 
    #            c = cluster.labels_) 
    # plt.xlabel('precios')
    # plt.ylabel('distancia')
    # plt.title('Clustering Jerárquico con k =' + str(k))
    # plt.legend(range(1, k + 1))

    # Calculamos el estadístico de sillhouete para ver cual es la mejor agrupación
    silluete.append(silhouette_score(hotelesNorm, cluster.labels_, metric = 'euclidean', 
                     random_state = 0))
    #printeamos el estadístico de sillhouete
    print("El estadístico de sillhouete para k = " + str(k) + " es: " + str(silluete[-1]))

# Graficamos el estadistico de la sillhouete
plt.plot(kden, silluete, '--bo', label = 'Sillhouette')

# Seleccionamos el mejor
#kOptima = kden[np.argmax(silluete)] #me coge 6
kOptima = 2 #est sillhouete o.431

# Calculamos la clasificación con el número k 
cluster = AgglomerativeClustering(n_clusters = kOptima, affinity = 'euclidean', 
                                  linkage = 'ward')  
# Predecimos el número de clases
cluster.fit_predict(hotelesNorm)

# Graficamos los valores de la mejor clasificación
plt.figure(figsize = (10, 7))  
plt.scatter(hotelesNorm['precios'], hotelesNorm['distancia'], 
            c = cluster.labels_) 
plt.xlabel('precios')
plt.ylabel('distancia')
plt.title('Clustering Jerárquico con k =' + str(kOptima))
plt.legend(range(0, kOptima))

# =============================================================================
###Adjusted Rand Index 
labels_pred = cluster.labels_
labels_true = hoteles['precios']
from sklearn.metrics.cluster import adjusted_rand_score
adjusted_rand_score(labels_true, labels_pred)
#El índice 0 nos indica que los dos clusters son significativamente diferentes. 
