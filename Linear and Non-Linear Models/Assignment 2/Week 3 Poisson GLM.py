# -*- coding: utf-8 -*-
"""
@name: Week 3 Poisson GLM.py
@author: Ming-Long Lam, Ph.D.
@organization: University of Chicago
"""
import matplotlib.pyplot as plt
import numpy
import pandas
import sys

# Set some options for printing all the columns
numpy.set_printoptions(precision = 10, threshold = sys.maxsize)
numpy.set_printoptions(linewidth = numpy.inf)

pandas.set_option('display.max_columns', None)
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', None)

pandas.options.display.float_format = '{:,.10}'.format

import seaborn
from scipy.stats import chi2

sys.path.append('C:\\MScAnalytics\\Linear and Nonlinear Model\\Job')
import Regression

target = 'CLM_COUNT'
exposure = 'EXPOSURE'
int_pred = ['HOMEKIDS', 'KIDSDRIV']

claim_history = pandas.read_excel('C:\\MScAnalytics\\Data\\claim_history.xlsx')

train_data = claim_history[claim_history['EXPOSURE'] > 0.0] # Only positive exposure
train_data = train_data[[target] + [exposure] + int_pred]   # Only necessary variables
train_data = train_data.dropna().reset_index()              # Remove missing values

# Missing value situation
print(train_data.isnull().sum())

# Display relationship in heatmap
for pred in int_pred:
    xtab = pandas.crosstab(train_data[target], train_data[pred], values = train_data[exposure], aggfunc=numpy.mean)
    xtab.fillna(0, inplace = True)
    print('\nPredictor: ', pred)
    print(xtab)

    plt.figure(figsize = (10,8), dpi = 200)
    ax = seaborn.heatmap(xtab, cmap = 'PiYG', cbar_kws = {'label': 'Mean Exposure'})
    ax.invert_yaxis()    
    plt.show()

n_sample = train_data.shape[0]
y_train = train_data[target]
o_train = numpy.log(train_data[exposure])

# Build a model with only the Intercept term
X_train = train_data[[target]]
X_train.insert(0, 'Intercept', 1.0)
X_train = X_train.drop(columns = target)

result = Regression.PoissonRegression (X_train, y_train, o_train)

outCoefficient = result[0]
outCovb = result[1]
outCorb = result[2]
llk = result[3]
nonAliasParam = result[4]
outIterationTable = result[5]
y_pred = result[6]

# Build the model Intercept + HOMEKIDS + KIDSDRIV
X_train = train_data[int_pred]
X_train.insert(0, 'Intercept', 1.0)

result = Regression.PoissonRegression (X_train, y_train, o_train)

outCoefficient = result[0]
outCovb = result[1]
outCorb = result[2]
llk = result[3]
nonAliasParam = result[4]
outIterationTable = result[5]
y_pred = result[6]

plt.figure(figsize = (8,4), dpi = 200)
ec = plt.scatter(y_train, y_pred, c = train_data['EXPOSURE'])
plt.xlabel('Number of Claims')
plt.ylabel('Predicted CLM_COUNT')
plt.xticks(range(10))
plt.grid(axis = 'both', linestyle = 'dotted')
plt.colorbar(ec, label = 'Exposure')
plt.show()

y_resid = y_train - y_pred
plt.figure(figsize = (8,4), dpi = 200)
ec = plt.scatter(y_train, y_resid, c = train_data['EXPOSURE'])
plt.xlabel('Number of Claims')
plt.ylabel('Simple Residual')
plt.xticks(range(10))
plt.grid(axis = 'both', linestyle = 'dotted')
plt.colorbar(ec, label = 'Exposure')
plt.show()
