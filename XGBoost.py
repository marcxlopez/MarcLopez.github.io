# -*- coding: utf-8 -*-
"""
Created on Thu May 12 10:34:35 2022
https://www.datacamp.com/tutorial/xgboost-in-python#hyperparameters
https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
@author: marcl
"""

import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV


# =============================================================================
# PARAMETROS MARC:
PATH = "C:\\Users\marcl\\Desktop\\TFG\\GITHUB TFG\\"
DATASETS_DIR = PATH + "data\\"
OUTPUT_DIR = PATH + "output\\"

#cargamos base de datos
hoteles = pd.read_pickle(DATASETS_DIR + 'HotelesImputados.pkl')

#==============================================================================
#separamos Y del resto de datos 
y = hoteles['precios']
X = hoteles.drop(['Hotel', 'ratioDescr','precios'], axis=1)

#Divide dataset intro TRAIN and TEST 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

###Regresor XGBOOST------------------------------------------------
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)

#Fit the regressor to the training set and make predictions on the test set
xg_reg.fit(X_train,y_train)
preds = xg_reg.predict(X_test)

#Compute the rmse by invoking the mean_sqaured_error function
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))

# -----------------------------------------------------------------------------
# Creamos una función para calcular 3 estados: 
#   - Acierto
#   - Fallo por abajo
#   - Fallo por encima


#==============================================================================
###k-fold Cross Validation using XGBoost
#to build more robust models, it is common to do a k-fold cross validation
# where all the entries in the original training dataset are used for both training as well as validation
xgb1 = xgb.XGBRegressor()
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'learning_rate': [.03, 0.05, .07], # so called `eta` value
              'max_depth': [5, 6, 7],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500]}

xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        cv = 10,
                        n_jobs = 5,
                        verbose = True)

xgb_grid.fit(X_train, y_train)

print(xgb_grid.best_score_)
print(xgb_grid.cv_results_)
print(xgb_grid.best_params_)

#==============================================================================
###Visualize Boosting Trees and Feature Importance
xg_reg = xgb.train(params=xgb_grid.best_params_, 
                   dtrain = xgb.DMatrix(data = X_train, label = y_train), 
                   num_boost_round = 100)

# -----------------------------------------------------------------------------
###examine the importance of each feature column in the original dataset
xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()

# -----------------------------------------------------------------------------
# Calculamos las predicciones finales 
#Fit the regressor to the training set and make predictions on the test set
preds = xg_reg.predict(xgb.DMatrix(data = X_test))

# -----------------------------------------------------------------------------
#Compute the rmse by invoking the mean_sqaured_error function
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))

# -----------------------------------------------------------------------------
# Creamos una función para calcular 3 estados: 
#   - Acierto
#   - Fallo por abajo
#   - Fallo por encima
factor = 0.1

dataset = pd.DataFrame({'ICinf': y_test - y_test*factor, 
                        'ICsup': y_test + y_test*factor})
dataset.reset_index(inplace=True)
dataset = dataset.drop(['index'], axis=1)

resultado = []
for i in range(0, len(y_test)):
    if dataset.loc[i, 'ICinf'] <= preds[i] and dataset.loc[i, 'ICsup'] >= preds[i]:
        resultado.append("OK")
    if dataset.loc[i, 'ICinf'] >= preds[i]:
        resultado.append("Subvalorado")
    if dataset.loc[i, 'ICsup'] <= preds[i]:
        resultado.append("Sobrevalorado")
    
def rel_freq(x):
    freqs = [(value, (x.count(value) / len(x))*100) for value in set(x)] 
    return freqs

tabla = pd.DataFrame(rel_freq(resultado), columns =['tipus', 'percent'])
print(tabla)

#==============================================================================
