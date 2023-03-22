# -*- coding: utf-8 -*-
"""
@Name: Week 7 Chicago Home Sale.py
@Creation Date: February 13, 2023
@author: Ming-Long Lam, Ph.D.
@organization: University of Chicago
(C) All Rights Reserved.
"""

import matplotlib.pyplot as plt
import numpy
import pandas
import sys

from scipy.stats import chi2

sys.path.append('C:\\MScAnalytics\\Linear and Nonlinear Model\\Job')
import Regression

# Set some options for printing all the columns
numpy.set_printoptions(precision = 10, threshold = sys.maxsize)
numpy.set_printoptions(linewidth = numpy.inf)

pandas.set_option('display.max_columns', None)
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', None)

pandas.options.display.float_format = '{:,.7}'.format

def PearsonCorrelation (x, y):
   '''Compute the Pearson correlation between two arrays x and y with the
   same number of values

   Argument:
   ---------
   x : a Pandas Series
   y : a Pandas Series
   
   Output:
   -------
   rho : Pearson correlation
   '''
   
   dev_x = x - numpy.mean(x)
   dev_y = y - numpy.mean(y)
   
   ss_xx = numpy.mean(dev_x * dev_x)
   ss_yy = numpy.mean(dev_y * dev_y)

   if (ss_xx > 0.0 and ss_yy > 0.0):
      ss_xy = numpy.mean(dev_x * dev_y)
      rho = (ss_xy / ss_xx) * (ss_xy / ss_yy)
      rho = numpy.sign(ss_xy) * numpy.sqrt(rho)
   else:
      rho = numpy.nan
      
   return (rho)

def RankOfValue (v):
   '''Compute the ranks of the values in an array v. For tied values, the
   average rank is computed.

   Argument:
   ---------
   v : a Pandas Series
   
   Output:
   -------
   rankv : Ranks of the values of v, minimum has a rank of zero
   '''

   uvalue, uinv, ucount = numpy.unique(v, return_inverse = True, return_counts = True)
   urank = []
   ur0 = 0
   for c in ucount:
      ur1 = ur0 + c - 1
      urank.append((ur0 + ur1)/2.0)
      ur0 = ur1 + 1

   rankv = []
   for j in uinv:
      rankv.append(urank[j])
      
   return (rankv)

def SpearmanCorrelation (x, y):
   '''Compute the Spearman rank-order correlation between two arrays x and y
   with the same number of values

   Argument:
   ---------
   x : a Pandas Series
   y : a Pandas Series
   
   Output:
   -------
   srho : Spearman rank-order correlation
   '''

   rank_x = RankOfValue(x)
   rank_y = RankOfValue(y)

   srho = PearsonCorrelation(rank_x, rank_y)
   return (srho)

def KendallTaub (x, y):
   '''Compute the Kendall's Tau-b correlation between two arrays x and y
   with the same number of values

   Argument:
   ---------
   x : a Pandas Series
   y : a Pandas Series
   
   Output:
   -------
   taub : Kendall's tau-b correlation
   '''

   nconcord = 0
   ndiscord = 0
   tie_x = 0
   tie_y = 0
   tie_xy = 0

   x_past = []
   y_past = []
   for xi, yi in zip(x, y):
      for xj, yj in zip(x_past, y_past):
         if (xi > xj):
            if (yi > yj):
               nconcord = nconcord + 1
            elif (yi < yj):
               ndiscord = ndiscord + 1
            else:
               tie_y = tie_y + 1
         elif (xi < xj):
            if (yi < yj):
               nconcord = nconcord + 1
            elif (yi > yj):
               ndiscord = ndiscord + 1
            else:
               tie_y = tie_y + 1
         else:
            if (yi == yj):
               tie_xy = tie_xy + 1
            else:
               tie_x = tie_x + 1

      x_past.append(xi)
      y_past.append(yi)

   denom = (nconcord + ndiscord + tie_x) * (nconcord + ndiscord + tie_y)
   if (denom > 0.0):
      taub = (nconcord - ndiscord) / numpy.sqrt(denom)
   else:
      taub = numpy.nan

   return (taub)

def AdjustedDistance (x):
   '''Compute the adjusted distances for an array x

   Argument:
   ---------
   x : a Pandas Series
   
   Output:
   -------
   adj_distance : Adjusted distances
   '''

   a_matrix = []
   row_mean = []

   for xi in x:
      a_row = numpy.abs(x - xi)
      row_mean.append(numpy.mean(a_row))
      a_matrix.append(a_row)
   total_mean = numpy.mean(row_mean)

   adj_m = []
   for row, rm in zip(a_matrix, row_mean):
      row = (row - row_mean) - (rm - total_mean)
      adj_m.append(row)

   return (numpy.array(adj_m))
   
def DistanceCorrelation (x, y):
   '''Compute the Distance correlation between two arrays x and y
   with the same number of values

   Argument:
   ---------
   x : a Pandas Series
   y : a Pandas Series
   
   Output:
   -------
   dcorr : Distance correlation
   '''

   adjD_x = AdjustedDistance (x)
   adjD_y = AdjustedDistance (y)

   v2sq_x = numpy.mean(numpy.square(adjD_x))
   v2sq_y = numpy.mean(numpy.square(adjD_y))
   v2sq_xy = numpy.mean(adjD_x * adjD_y)
   
   if (v2sq_x > 0.0 and v2sq_y > 0.0):
      dcorr = (v2sq_xy / v2sq_x) * (v2sq_xy / v2sq_y)
      dcorr = numpy.power(dcorr, 0.25)

   return (dcorr)

catName = ['Township Name', "O'Hare Noise Indicator", 'FEMA Floodplain', 'Road Proximity < 100 Feet', 'Road Proximity 101 - 300 Feet',
           'Garage 1 Size', 'Sale Year', 'Sale Month']
intName = ['Land Acre', 'Building Square Feet', 'Age', 'Bedrooms', 'Full Baths', 'Half Baths', 'Tract Median Income', 'Tax Rate']

nPredictor = len(catName) + len(intName)

yName = 'Sale Price'

CookHomeSale = pandas.read_csv('C:\\MScAnalytics\\Data\\ChicagoHomeSale.csv')
print(CookHomeSale.columns)

# Manage a few features
CookHomeSale['Sale Price'] = CookHomeSale['Sale Price'] / 1000.0
CookHomeSale['Tract Median Income'] = CookHomeSale['Tract Median Income'] / 1000.0
CookHomeSale['Land Acre'] = CookHomeSale['Land Square Feet'] / 43560.0
CookHomeSale['Building Square Feet'] = CookHomeSale['Building Square Feet'] / 1000.0

dateObj = pandas.to_datetime(CookHomeSale['Sale Date'], format = '%m/%d/%Y')
CookHomeSale['Sale Year'] = dateObj.apply(lambda x: x.year)
CookHomeSale['Sale Month'] = dateObj.apply(lambda x: x.month)

# Explore the Use variable
fig, ax0 = plt.subplots(nrows = 1, ncols = 1, dpi = 200, sharey = True, figsize = (6,4))
ufreq = CookHomeSale['Use'].astype('category').value_counts()
print(ufreq)
ax0.bar(ufreq.index, ufreq, color = 'royalblue')
ax0.set_xlabel('Usage of Property')
ax0.set_ylabel('Number of Observations')
ax0.set_yticks(range(0, 40000, 5000))
ax0.yaxis.grid(True)
plt.show()

# Focus on only Single-Family usage
trainData = CookHomeSale[CookHomeSale['Use'] == 'Single-Family'].reset_index(drop = True)

# Explore the Sale Year and Sale Month variables
fig, (ax0, ax1) = plt.subplots(nrows = 1, ncols = 2, dpi = 200, figsize = (12,4))
ufreq = trainData['Sale Year'].astype('category').value_counts()
ax0.bar(ufreq.index, ufreq, color = 'royalblue')
ax0.set_xlabel('Sale Year of Property')
ax0.set_ylabel('Number of Observations')
ax0.set_xticks([2018, 2019, 2020])
ax0.set_yticks(range(0, 14000, 2000))
ax0.yaxis.grid(True)

ufreq = trainData['Sale Month'].astype('category').value_counts()
ax1.bar(ufreq.index, ufreq, color = 'teal')
ax1.set_xlabel('Sale Month of Property')
ax1.set_ylabel('Number of Observations')
ax1.set_xticks(range(1,13,1))
ax1.set_yticks(range(0, 5000, 1000))
ax1.yaxis.grid(True)
plt.show()

# Explore the Sale Price variable
fig, (ax0, ax1) = plt.subplots(nrows = 2, ncols = 1, dpi = 200, sharex = True, figsize = (12,6),
                               gridspec_kw = {'height_ratios': [2, 1]})
ax0.hist(trainData[yName], bins = numpy.arange(0.0, 7500.0, 500.0), color = 'royalblue')
ax0.set_xlabel('')
ax0.set_ylabel('Number of Observations')
ax0.set_yticks(range(0, 35000, 5000))
ax0.yaxis.grid(True)

trainData.boxplot(column = yName, ax = ax1, vert = False, figsize = (12,2))
ax1.set_xlabel('Sale Price for Home (in Thousands of Dollars)')
ax1.set_ylabel('')
ax1.set_xticks(numpy.arange(0.0, 7500.0, 500.0))
ax1.xaxis.grid(True)
plt.suptitle('')
plt.title('')
plt.show()

print(trainData[[yName]].describe())

# Explore the categorical predictors using grouped boxplot
for X_name in catName:
    print(trainData[[X_name]].value_counts())

    fig, ax1 = plt.subplots(nrows = 1, ncols = 1, dpi = 200)
    trainData.boxplot(column = yName, by = X_name, ax = ax1, vert = False)
    ax1.set_xlabel('Sale Price for Home (in Thousands of Dollars)')
    ax1.xaxis.grid(True)
    ax1.invert_yaxis()
    plt.suptitle('')
    plt.title(X_name)
    plt.show()

# Explore the continuous predictors using scatterplot
for X_name in intName:
    print(trainData[[X_name]].describe())

    fig, ax1 = plt.subplots(nrows = 1, ncols = 1, dpi = 200, figsize = (7,5))
    ax1.scatter(trainData[X_name], trainData[yName], color = 'royalblue')
    ax1.set_xlabel(X_name)
    ax1.set_ylabel('Sale Price for Home (in Thousands of Dollars)')
    ax1.set_yticks(numpy.arange(0.0, 8000.0, 1000.0))
    ax1.xaxis.grid(True)
    ax1.yaxis.grid(True)
    plt.show()

# Reorder the categories of the categorical variables in ascending frequency
for pred in catName:
    u = trainData[pred].astype('category').copy()
    u_freq = u.value_counts(ascending = True)
    trainData[pred] = u.cat.reorder_categories(list(u_freq.index)).copy()

# Generate a column of Intercept
X0_train = trainData[[yName]].copy()
X0_train.insert(0, 'Intercept', 1.0)
X0_train.drop(columns = [yName], inplace = True)

y_train = trainData[yName].copy()

maxIter = 20
tolS = 1e-7
stepSummary = []

# Intercept only model
resultList = Regression.GammaRegression (X0_train, y_train, offset = None, maxIter = maxIter, tolSweep = tolS)

llk0 = resultList[3]
df0 = len(resultList[4])
stepSummary.append([0, 'Intercept', ' ', 0, df0, llk0, numpy.NaN, numpy.NaN, numpy.NaN])

cName = catName.copy()
iName = intName.copy()
entryThreshold = 0.05

# The Deviance significance is the sixth element in each row of the test result
def takeDevSig(s):
    return s[7]

for step in range(nPredictor):
    enterName = ''
    stepDetail = []

    # Enter the next predictor
    for X_name in cName:
        X_train = X0_train.join(pandas.get_dummies(trainData[[X_name]]))
        resultList = Regression.GammaRegression (X_train, y_train, offset = None, maxIter = maxIter, tolSweep = tolS)
        llk1 = resultList[3]
        df1 = len(resultList[4])
        n_iter = resultList[5].shape[0]
        devChiSq = 2.0 * (llk1 - llk0)
        devDF = df1 - df0
        devSig = chi2.sf(devChiSq, devDF)
        stepDetail.append([X_name, 'categorical', n_iter, df1, llk1, devChiSq, devDF, devSig])

    for X_name in iName:
        X_train = X0_train.join(trainData[[X_name]])
        resultList = Regression.GammaRegression (X_train, y_train, offset = None, maxIter = maxIter, tolSweep = tolS)
        llk1 = resultList[3]
        df1 = len(resultList[4])
        n_iter = resultList[5].shape[0]
        devChiSq = 2.0 * (llk1 - llk0)
        devDF = df1 - df0
        devSig = chi2.sf(devChiSq, devDF)
        stepDetail.append([X_name, 'interval', n_iter, df1, llk1, devChiSq, devDF, devSig])

    # Find a predictor to add, if any
    stepDetail.sort(key = takeDevSig, reverse = False)
    minSig = takeDevSig(stepDetail[0])
    if (minSig <= entryThreshold):
        add_var = stepDetail[0][0]
        add_type = stepDetail[0][1]
        df0 = stepDetail[0][3]
        llk0 = stepDetail[0][4]
        stepSummary.append([step+1] + stepDetail[0])
        if (add_type == 'categorical'):
            X0_train = X0_train.join(pandas.get_dummies(trainData[[add_var]].astype('category')))
            cName.remove(add_var)
        else:
            X0_train = X0_train.join(trainData[[add_var]])
            iName.remove(add_var)
    else:
        break


    # Print debugging output
    print('======= Step Detail =======')
    print('Step = ', step+1)
    print('Step Statistics:')
    print(pandas.DataFrame(stepDetail, columns = ['Predictor', 'Type', 'N Iter', 'ModelDF', 'ModelLLK', 'DevChiSq', 'DevDF', 'DevSig']))
    print('Enter predictor = ', add_var)
    print('Minimum P-Value =', minSig)
    print('\n')

# End of forward selection
print('======= Step Summary =======')
stepSummary = pandas.DataFrame(stepSummary, columns = ['Step', 'Predictor', 'Type', 'N Iteration', 'ModelDF', 'ModelLLK', 'DevChiSq', 'DevDF', 'DevSig'])
print(stepSummary)

# Final model
resultList = Regression.GammaRegression (X0_train, y_train, offset = None, maxIter = maxIter, tolSweep = tolS)

outCoefficient = resultList[0]
alpha = resultList[7]
y_pred = resultList[6]

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
pearson_corr = PearsonCorrelation (y_train, y_pred)
spearman_corr = SpearmanCorrelation (y_train, y_pred)
kendall_tau = KendallTaub (y_train, y_pred)
distance_corr = DistanceCorrelation (y_train, y_pred)

rsq = pearson_corr * pearson_corr

# Pearson Residual
y_pearson_residual = y_simple_residual / numpy.sqrt(y_pred)

# Deviance Residual
r_vec = y_train / y_pred
di_2 = 2 * (r_vec - numpy.log(r_vec) - 1)
y_deviance_residual = numpy.where(y_simple_residual > 0, 1.0, -1.0) * numpy.sqrt(di_2)

fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(nrows = 2, ncols = 2, dpi = 200, sharex = True,
                                             figsize = (24,12))

# Plot predicted sale price versus observed sale price
ax0.scatter(y_train, y_pred, c = 'royalblue', marker = 'o')
ax0.set_xlabel('')
ax0.set_ylabel('Predicted Sale Price (thousand dollars)')
ax0.xaxis.grid(True)
ax0.yaxis.grid(True)

# Plot simple residuals versus observed sale price
ax1.scatter(y_train, y_simple_residual, c = 'royalblue', marker = 'o')
ax1.set_xlabel('')
ax1.set_ylabel('Simple Residual')
ax1.xaxis.grid(True)
ax1.yaxis.grid(True)

# Plot Pearson residuals versus observed sale price
ax2.scatter(y_train, y_pearson_residual, c = 'royalblue', marker = 'o')
ax2.set_xlabel('Observed Sale Price (thousand dollars)')
ax2.set_ylabel('Pearson Residual')
ax2.xaxis.grid(True)
ax2.yaxis.grid(True)

# Plot deviance residuals versus observed sale price
ax3.scatter(y_train, y_deviance_residual, c = 'royalblue', marker = 'o')
ax3.set_xlabel('Observed Sale Price (thousand dollars)')
ax3.set_ylabel('Deviance Residual')
ax3.xaxis.grid(True)
ax3.yaxis.grid(True)

plt.show()

# Plot absolute proportion error versus actual value
fig, ax0 = plt.subplots(nrows = 1, ncols = 1, dpi = 200, figsize = (10,6))
ax0.scatter(y_train, ape, c = 'royalblue', marker = 'o')
ax0.set_xlabel('Sale Price (thousand dollars)')
ax0.set_ylabel('Absolute Proportion Error')
ax0.xaxis.grid(True)
ax0.yaxis.grid(True)

plt.figure(dpi = 200, figsize = (10,6))
sg = plt.scatter(y_train, ape, c = y_train, s = 10, marker = 'o')
plt.xlabel('Observed Sale Price (thousand dollar)')
plt.ylabel('Absolute Proportion Error')
plt.yscale('log')
plt.grid(axis = 'both')
cbar = plt.colorbar(sg, label = 'Predicted Sales Price (thousand dollar)')
cbar.set_ticks(numpy.arange(0.0, 7000.0, 1000.0))
plt.show()

# Identify extremely high predictions
# Explore the Sale Price variable
fig, (ax0, ax1) = plt.subplots(nrows = 2, ncols = 1, dpi = 200, sharex = True, figsize = (12,5),
                               gridspec_kw = {'height_ratios': [1, 1]})
trainData.boxplot(column = yName, ax = ax0, vert = False, figsize = (12,2))
ax0.set_xlabel('Observed Sale Price for Home (in Thousands of Dollars)')
ax0.set_ylabel('')
ax0.set_xticks(numpy.arange(0.0, 25000.0, 2500.0))
ax0.xaxis.grid(True)

plotData = pandas.DataFrame(y_pred, columns = [yName])
plotData.boxplot(column = yName, ax = ax1, vert = False, figsize = (12,2))
ax1.set_xlabel('Predicted Sale Price for Home (in Thousands of Dollars)')
ax1.set_ylabel('')
ax1.set_xticks(numpy.arange(0.0, 25000.0, 2500.0))
ax1.xaxis.grid(True)
plt.suptitle('')
plt.title('')

plt.show()
