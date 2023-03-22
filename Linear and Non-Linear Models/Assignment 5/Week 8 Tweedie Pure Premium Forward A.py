# -*- coding: utf-8 -*-
"""
@Name: Week 8 Tweedie Pure Premium Forward A.py
@Creation Date: February 20, 2023
@author: Ming-Long Lam, Ph.D.
@organization: University of Chicago
(C) All Rights Reserved.
"""

import matplotlib.pyplot as plt
import numpy
import pandas
import sys

from scipy.stats import chi2

# sys.path.append('C:\\MScAnalytics\\Linear and Nonlinear Model\\Job')
import Regression

# Set some options for printing all the columns
numpy.set_printoptions(precision = 10, threshold = sys.maxsize)
numpy.set_printoptions(linewidth = numpy.inf)

pandas.set_option('display.max_columns', None)
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', None)

pandas.options.display.float_format = '{:,.7f}'.format

catName = ['CAR_TYPE']
intName = ['CAR_AGE', 'HOMEKIDS', 'KIDSDRIV']
yName = 'CLM_AMT'
eName = 'EXPOSURE'

claim_history = pandas.read_excel('claim_history.xlsx')

train_data = claim_history[[yName, eName] + catName + intName]
train_data = train_data[train_data[eName] > 0.0].dropna().reset_index(drop = True)

y_train = train_data[yName]
o_train = numpy.log(train_data[eName])

# Histogram of Claim Amount
plt.figure(figsize = (10,6), dpi = 200)
plt.hist(y_train, bins = numpy.arange(0,13000,500), fill = True, color = 'royalblue', edgecolor = 'black')
plt.title('Original Y-Scale')
plt.xlabel('Claim Amount (Dollar)')
plt.ylabel('Number of Observations')
plt.xticks(numpy.arange(0,14000,2000))
plt.grid(axis = 'y', linewidth = 0.7, linestyle = 'dashed')
plt.show()

plt.figure(figsize = (10,6), dpi = 200)
plt.hist(y_train, bins = numpy.arange(0,13000,500), fill = True, color = 'lightyellow', edgecolor = 'black')
plt.title('Base 10 Logarithm Y-Scale')
plt.xlabel('Claim Amount (Dollar)')
plt.ylabel('Number of Observations')
plt.xticks(numpy.arange(0,14000,2000))
plt.yscale('log')
plt.grid(axis = 'y', linewidth = 0.7, linestyle = 'dashed')
plt.show()

# Estimate the Tweedie's P value
xtab = pandas.pivot_table(train_data, values = yName, index = catName + intName,
                          columns = None, aggfunc = ['count', 'mean', 'var'])

cell_stats = xtab[['mean','var']].reset_index().droplevel(1, axis = 1)

ln_Mean = numpy.where(cell_stats['mean'] > 1e-16, numpy.log(cell_stats['mean']), numpy.NaN)
ln_Variance = numpy.where(cell_stats['var'] > 1e-16, numpy.log(cell_stats['var']), numpy.NaN)

use_cell = numpy.logical_not(numpy.logical_or(numpy.isnan(ln_Mean), numpy.isnan(ln_Variance)))

X_train = ln_Mean[use_cell]
y_train = ln_Variance[use_cell]

# Scatterplot of lnVariance vs lnMean
plt.figure(figsize = (8,6), dpi = 200)
plt.scatter(X_train, y_train, c = 'royalblue')
plt.xlabel('Log(Mean)')
plt.ylabel('Log(Variance)')
plt.margins(0.1)
plt.grid(axis = 'both', linewidth = 0.7, linestyle = 'dashed')
plt.show()

X_train = pandas.DataFrame(X_train, columns = ['ln_Mean'])
X_train.insert(0, 'Intercept', 1.0)

y_train = pandas.Series(y_train, name = 'ln_Variance')

result_list = Regression.LinearRegression (X_train, y_train)

tweediePower = result_list[0][1]
tweediePhi = numpy.exp(result_list[0][0])

# Begin Forward Selection

# The Deviance significance is the sixth element in each row of the test result
def takeDevSig(s):
    return s[7]

nPredictor = len(catName) + len(intName)
stepSummary = []

# Intercept only model
X0_train = train_data[[]]
X0_train.insert(0, 'Intercept', 1.0)

y_train = train_data[yName]

result_list = Regression.TweedieRegression (X0_train, y_train, o_train, tweedieP = tweediePower)
qllk0 = result_list[3]
df0 = len(result_list[4])
phi0 = result_list[7]

stepSummary.append([0, 'Intercept', ' ', df0, qllk0, phi0, numpy.NaN, numpy.NaN, numpy.NaN])

cName = catName.copy()
iName = intName.copy()
entryThreshold = 0.05

for step in range(nPredictor):
   enterName = ''
   stepDetail = []

   # Enter the next predictor
   for X_name in cName:
      X_train = pandas.get_dummies(train_data[[X_name]].astype('category'))
      if (X0_train is not None):
         X_train = X0_train.join(X_train)
      result_list = Regression.TweedieRegression (X_train, y_train, o_train, tweedieP = tweediePower)
      qllk1 = result_list[3]
      df1 = len(result_list[4])
      phi1 = result_list[7]
      devChiSq = 2.0 * (qllk1 - qllk0) / phi0
      devDF = df1 - df0
      devPValue = chi2.sf(devChiSq, devDF)
      stepDetail.append([X_name, 'categorical', df1, qllk1, phi1, devChiSq, devDF, devPValue])

   for X_name in iName:
      X_train = train_data[[X_name]]
      if (X0_train is not None):
         X_train = X0_train.join(X_train)
      result_list = Regression.TweedieRegression (X_train, y_train, o_train, tweedieP = tweediePower)
      qllk1 = result_list[3]
      df1 = len(result_list[4])
      phi1 = result_list[7]
      devChiSq = 2.0 * (qllk1 - qllk0) / phi0
      devDF = df1 - df0
      devPValue = chi2.sf(devChiSq, devDF)
      stepDetail.append([X_name, 'interval', df1, qllk1, phi1, devChiSq, devDF, devPValue])

   # Find a predictor to add, if any
   stepDetail.sort(key = takeDevSig, reverse = False)
   minSig = takeDevSig(stepDetail[0])
   if (minSig <= entryThreshold):
      add_var = stepDetail[0][0]
      add_type = stepDetail[0][1]
      df0 = stepDetail[0][2]
      qllk0 = stepDetail[0][3]
      phi0 = stepDetail[0][4]
      stepSummary.append([step+1] + stepDetail[0])
      if (add_type == 'categorical'):
         X0_train = X0_train.join(pandas.get_dummies(train_data[[add_var]].astype('category')))
         cName.remove(add_var)
      else:
         X0_train = X0_train.join(train_data[[add_var]])
         iName.remove(add_var)
   else:
        break

# End of forward selection

stepSummary_df = pandas.DataFrame(stepSummary, columns = ['Step','Predictor','Type','N Non-Aliased Parameters',
                                                          'Quasi Log-Likelihood', 'Phi', 'Deviance ChiSquare',
                                                          'Deviance DF', 'Deviance Sig.'])
print(stepSummary_df)
# Retrain the final model
result_list = Regression.TweedieRegression (X0_train, y_train, o_train, tweedieP = tweediePower)
y_pred = result_list[6]

# Simple Residual
y_simple_residual = y_train - y_pred

# Mean Absolute Proportion Error
ape = numpy.abs(y_simple_residual) / y_train
mape = numpy.mean(ape)

# Root Mean Squared Error
mse = numpy.mean(numpy.power(y_simple_residual, 2))
rmse = numpy.sqrt(mse)

# Relative Error
relerr = mse / numpy.var(y_train, ddof = 0)

# R-Squared
pearson_corr = Regression.PearsonCorrelation (y_train, y_pred)
spearman_corr = Regression.SpearmanCorrelation (y_train, y_pred)
kendall_tau = Regression.KendallTaub (y_train, y_pred)
distance_corr = Regression.DistanceCorrelation (y_train, y_pred)

idx_positive = (y_train > 0.0)
y_train_pos = y_train[idx_positive]
y_pred_pos = y_pred[idx_positive]

# Simple Residual
y_simple_residual_pos = y_train_pos - y_pred_pos

# Mean Absolute Proportion Error
ape = numpy.abs(y_simple_residual_pos) / y_train_pos
mape_pos = numpy.mean(ape)

# Root Mean Squared Error
mse = numpy.mean(numpy.power(y_simple_residual_pos, 2))
rmse_pos = numpy.sqrt(mse)

# Relative Error
relerr_pos = mse / numpy.var(y_train_pos, ddof = 0)

pearson_corr_pos = Regression.PearsonCorrelation (y_train_pos, y_pred_pos)
spearman_corr_pos = Regression.SpearmanCorrelation (y_train_pos, y_pred_pos)
kendall_tau_pos = Regression.KendallTaub (y_train_pos, y_pred_pos)
distance_corr_pos = Regression.DistanceCorrelation (y_train_pos, y_pred_pos)

e_train =  train_data[eName]

# Plot the predicted Number of Claims versus the observed Number of Claims

fig, ax = plt.subplots(1, 1, figsize = (15,6), dpi = 200)
sg = ax.scatter(y_train, y_pred, c = e_train, s = 20, marker = 'o')
ax.set_xlabel('Observed Claim Amount')
ax.set_ylabel('Predicted Claim Amount')
ax.xaxis.set_major_formatter('${x:,.0f}')
ax.yaxis.set_major_formatter('${x:,.0f}')
ax.grid(axis = 'both')
plt.colorbar(sg, label = 'Exposure')
plt.show()

# Plot the predicted Number of Claims versus the observed Number of Claims
obs_pp = y_train / e_train
pred_pp = y_pred / e_train

fig, ax = plt.subplots(1, 1, figsize = (15,6), dpi = 200)
sg = ax.scatter(obs_pp, pred_pp, c = e_train, s = 20, marker = 'o')
ax.set_xlabel('Observed Pure Premium')
ax.set_ylabel('Predicted Pure Premium')
ax.xaxis.set_major_formatter('${x:,.0f}')
ax.yaxis.set_major_formatter('${x:,.0f}')
ax.grid(axis = 'both')
plt.colorbar(sg, label = 'Exposure')
plt.show()

fig, ax = plt.subplots(1, 1, figsize = (15,6), dpi = 200)
sg = ax.scatter(obs_pp, pred_pp, c = e_train, s = 20, marker = 'o')
ax.set_xlabel('Observed Pure Premium Less than $20,000')
ax.set_ylabel('Predicted Pure Premium')
ax.set_xlim(0, 20000)
ax.xaxis.set_major_formatter('${x:,.0f}')
ax.yaxis.set_major_formatter('${x:,.0f}')
ax.grid(axis = 'both')
plt.colorbar(sg, label = 'Exposure')
plt.show()


