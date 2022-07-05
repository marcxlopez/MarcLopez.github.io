# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 13:01:39 2022
https://www.cienciadedatos.net/documentos/py10-regresion-lineal-python.html
@author: marcl
"""
# Tratamiento de datos
# ==============================================================================
import pandas as pd
import numpy as np


# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

# Preprocesado y análisis
# ==============================================================================
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import pingouin as pg
from scipy import stats as stats 
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from scipy import stats
# Configuración matplotlib
# ==============================================================================
plt.style.use('ggplot')

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

## PARAMETROS MARC:
PATH = "C:\\Users\marcl\\Desktop\\TFG\\GITHUB TFG\\"
DATASETS_DIR = PATH + "data\\"
hoteles = pd.read_pickle(DATASETS_DIR + 'HotelesImputados.pkl')
hoteles = hoteles.dropna(subset=['precios'], axis=0)

####### Correlación lineal entre las dos variables
# ==============================================================================

corr_test = pearsonr(x =hoteles['Estrellas'], y = hoteles['precios'])
print("Coeficiente de correlación de Pearson: ", corr_test[0])
print("P-value: ", corr_test[1])

corr_test = pearsonr(x =hoteles['distancia'], y = hoteles['precios'])
print("Coeficiente de correlación de Pearson: ", corr_test[0])
print("P-value: ", corr_test[1])

corr_test = pearsonr(x =hoteles['Prox_Dalt Vila'], y = hoteles['precios'])
print("Coeficiente de correlación de Pearson: ", corr_test[0]) 
print("P-value: ", corr_test[1])

corr_test = pearsonr(x =hoteles['Prox_Sant Joan de Labritja'], y = hoteles['precios'])
print("Coeficiente de correlación de Pearson: ", corr_test[0]) 
print("P-value: ", corr_test[1])

corr_test = pearsonr(x =hoteles["Prox_Cala d'Hort"], y = hoteles['precios'])
print("Coeficiente de correlación de Pearson: ", corr_test[0]) 
print("P-value: ", corr_test[1])

corr_test = pearsonr(x =hoteles['Prox_Cala de Sant Vicent'], y = hoteles['precios'])
print("Coeficiente de correlación de Pearson: ", corr_test[0]) 
print("P-value: ", corr_test[1])

corr_test = pearsonr(x =hoteles['Prox_Cala Comte'], y = hoteles['precios'])
print("Coeficiente de correlación de Pearson: ", corr_test[0]) 
print("P-value: ", corr_test[1])

corr_test = pearsonr(x =hoteles['Prox_Aeropuerto de Ibiza'], y = hoteles['precios'])
print("Coeficiente de correlación de Pearson: ", corr_test[0]) 
print("P-value: ", corr_test[1])

corr_test = pearsonr(x =hoteles['Prox_Ses Salines'], y = hoteles['precios'])
print("Coeficiente de correlación de Pearson: ", corr_test[0]) 
print("P-value: ", corr_test[1])
corr_test = pearsonr(x =hoteles['Prox_Cala Benirrás'], y = hoteles['precios'])
print("Coeficiente de correlación de Pearson: ", corr_test[0]) 
print("P-value: ", corr_test[1])

corr_test = pearsonr(x = hoteles.drop(['Hotel', 'ratioDescr','precios'], y = hoteles['precios'])
print("Coeficiente de correlación de Pearson: ", corr_test[0]) 
print("P-value: ", corr_test[1])

#Observamos que la proximidad a Ses salines es NO ES SIGNIFICATIVA para el modelo. En cambio, la variable distancia SI lo es. 
#Variables demasiado correlacionadas

##Gráfico de NORMALIDAD 
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

axs[0].hist(x=hoteles.distancia, bins=20, color="#3182bd", alpha=0.5)
axs[0].plot(hoteles.distancia, np.full_like(hoteles.distancia, -0.01), '|k', markeredgewidth=1)
axs[0].set_title('Distribución distancias')
axs[0].set_xlabel('Distancias')
axs[0].set_ylabel('counts')

axs[1].hist(x=hoteles.precios, bins=20, color="#3182bd", alpha=0.5)
axs[1].plot(hoteles.precios, np.full_like(hoteles.precios, -0.01), '|k', markeredgewidth=1)
axs[1].set_title('Distribución precio')
axs[1].set_xlabel('precio')
axs[1].set_ylabel('counts')

# Normalidad de los residuos Shapiro-Wilk test
# ==============================================================================
shapiro_test = stats.shapiro(hoteles.precios)
print(f"Variable height: {shapiro_test}")
shapiro_test = stats.shapiro(hoteles.distancia)
print(f"Variable weight: {shapiro_test}")
shapiro_test = stats.shapiro(hoteles.Estrellas)
print(f"Variable height: {shapiro_test}")
plt.tight_layout();

###############################################################################

# División de los datos en train y test
#separamos Y del resto de datos 
y = hoteles['precios']
X = hoteles.drop(['Hotel', 'ratioDescr','precios'], axis=1)

#Divide dataset intro TRAIN and TEST 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
# ------------------------------------------------------------------------------

# ==============================================================================
modelo = LinearRegression()
modelo.fit(X = X_train, y = y_train) #fallo
# Información del modelo
# ==============================================================================
print("Intercept:", modelo.intercept_)
#print("Coeficiente:", list(zip(X.columns, modelo.coef_.flatten(), )))
print("Coeficiente de determinación R^2:", modelo.score(X, y))
# Error de test del modelo 
# ==============================================================================
predicciones = modelo.predict(X = X_test)
print(predicciones[0:3,])

rmse = mean_squared_error(
        y_true  = y_test,
        y_pred  = predicciones,
        squared = False
       )
print("")
print(f"El error (rmse) de test es: {rmse}")

###############################################################################

###############################################################################
# Diagnóstico errores (residuos) de las predicciones de entrenamiento
# ==============================================================================
#y_train = y_train.flatten()
prediccion_train = modelo.predict(X_train)
residuos_train   = prediccion_train - y_train
#Inspección visual
# Gráficos
# ==============================================================================
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 8))

axes[0, 0].scatter(y_train, prediccion_train, edgecolors=(0, 0, 0), alpha = 0.4)
axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],
                'k--', color = 'black', lw=2)
axes[0, 0].set_title('Valor predicho vs valor real', fontsize = 10, fontweight = "bold")
axes[0, 0].set_xlabel('Real')
axes[0, 0].set_ylabel('Predicción')
axes[0, 0].tick_params(labelsize = 7)

axes[0, 1].scatter(list(range(len(y_train))), residuos_train,
                   edgecolors=(0, 0, 0), alpha = 0.4)
axes[0, 1].axhline(y = 0, linestyle = '--', color = 'black', lw=2)
axes[0, 1].set_title('Residuos del modelo', fontsize = 10, fontweight = "bold")
axes[0, 1].set_xlabel('id')
axes[0, 1].set_ylabel('Residuo')
axes[0, 1].tick_params(labelsize = 7)

sns.histplot(
    data    = residuos_train,
    stat    = "density",
    kde     = True,
    line_kws= {'linewidth': 1},
    color   = "firebrick",
    alpha   = 0.3,
    ax      = axes[1, 0]
)

axes[1, 0].set_title('Distribución residuos del modelo', fontsize = 10,
                     fontweight = "bold")
axes[1, 0].set_xlabel("Residuo")
axes[1, 0].tick_params(labelsize = 7)


sm.qqplot(
    residuos_train,
    fit   = True,
    line  = 'q',
    ax    = axes[1, 1], 
    color = 'firebrick',
    alpha = 0.4,
    lw    = 2
)
axes[1, 1].set_title('Q-Q residuos del modelo', fontsize = 10, fontweight = "bold")
axes[1, 1].tick_params(labelsize = 7)

axes[2, 0].scatter(prediccion_train, residuos_train,
                   edgecolors=(0, 0, 0), alpha = 0.4)
axes[2, 0].axhline(y = 0, linestyle = '--', color = 'black', lw=2)
axes[2, 0].set_title('Residuos del modelo vs predicción', fontsize = 10, fontweight = "bold")
axes[2, 0].set_xlabel('Predicción')
axes[2, 0].set_ylabel('Residuo')
axes[2, 0].tick_params(labelsize = 7)

# Se eliminan los axes vacíos
fig.delaxes(axes[2,1])

fig.tight_layout()
plt.subplots_adjust(top=0.9)
fig.suptitle('Diagnóstico residuos', fontsize = 12, fontweight = "bold");

# Normalidad de los residuos Shapiro-Wilk test
# ==============================================================================
shapiro_test = stats.shapiro(residuos_train)
shapiro_test