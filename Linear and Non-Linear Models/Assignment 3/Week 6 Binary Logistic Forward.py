# -*- coding: utf-8 -*-
"""
@author: Ming-Long Lam, Ph.D.
@organization: University of Chicago
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

pandas.options.display.float_format = '{:,.7e}'.format

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)

catName = ['primary_age_tier', 'primary_gender', 'marital', 'residence_location', 'aoi_tier']
intName = []
yName = 'claim_indicator'

nPredictor = len(catName) + len(intName)

# Create the Claim Indicator (1 if there are claims, and 0 otherwise)
claim = pandas.read_excel('C:\\MScAnalytics\\Data\\HO_claim_history.xlsx')
claim[yName] = claim['num_claims'].apply(lambda x: 1 if x > 0 else 0)

# Frequency of the nominal target
fig, (ax0, ax1) = plt.subplots(nrows = 1, ncols = 2, dpi = 200, figsize = (10,4))
plotData = claim['num_claims'].value_counts(ascending = False)
plotData.sort_index(inplace = True)
ax0.bar(plotData.index, plotData, color = 'royalblue')
ax0.set_xlabel('Number of Claims')
ax0.set_ylabel('Number of Observations')
ax0.set_xticks(plotData.index)
ax0.grid(axis = 'y')

plotData = claim[yName].value_counts(ascending = False)
plotData.sort_index(inplace = True)
ax1.bar(plotData.index, plotData, color = 'lightgreen')
ax1.set_xlabel('Claim Indicator')
ax1.set_ylabel('')
ax1.set_xticks([0,1])
ax1.grid(axis = 'y')
plt.show()

print('=== Frequency of ' + yName + ' ===')
print(plotData)

# Create the training data
trainData = claim[[yName] + catName + intName].dropna()
del claim

# Arrange categories of particular ordinal predictor
pred = 'primary_age_tier'
u = trainData[pred].astype('category').copy()
trainData[pred] = u.cat.reorder_categories(['< 21','21 - 27','28 - 37','38 - 60','> 60']).copy()

pred = 'marital'
u = trainData[pred].astype('category').copy()
trainData[pred] = u.cat.reorder_categories(['Not Married','Married','Un-Married']).copy()

pred = 'aoi_tier'
u = trainData[pred].astype('category').copy()
trainData[pred] = u.cat.reorder_categories(['< 100K','100K - 350K','351K - 600K','601K - 1M','> 1M']).copy()

n_sample = trainData.shape[0]

# Specify the color sequence
cmap = ['indianred', 'royalblue']

# Explore categorical predictor
if (len(catName) > 0):
    print('=== Frequency of  Categorical Predictors ===')
    print(trainData[catName].value_counts())

for pred in catName:

    # Generate the contingency table of the categorical input feature by the target
    cntTable = pandas.crosstab(index = trainData[pred], columns = trainData[yName], margins = False, dropna = True)

    # Calculate the row percents
    pctTable = 100.0 * cntTable.div(cntTable.sum(1), axis = 'index')

    fig, (ax0, ax1, ax2) = plt.subplots(nrows = 1, ncols = 3, dpi = 200, figsize = (18,6))

    # Generate a horizontal stacked percentage bar chart
    barThick = 0.8
    yCat = cntTable.columns
    accPct = numpy.zeros(pctTable.shape[0])
    for j in range(len(yCat)):
        catLabel = yCat[j]
        ax0.barh(pctTable.index, pctTable[catLabel], color = cmap[j], left = accPct, label = catLabel, height = barThick)
        accPct = accPct + pctTable[catLabel]
    ax0.xaxis.set_major_locator(MultipleLocator(base = 20))
    ax0.xaxis.set_major_formatter(FormatStrFormatter('%.0f%%'))
    ax0.xaxis.set_minor_locator(MultipleLocator(base = 5))
    ax0.set_xlabel('Percent')
    ax0.set_ylabel(pred)
    ax0.grid(axis = 'x', linestyle = '--', linewidth = 0.5)
    ax0.legend(loc = 'lower center', bbox_to_anchor = (0.35, 1), ncol = 3)

    allTable = cntTable[0] + cntTable[1]
    oddsTable = cntTable[1] / cntTable[0]

    ax1.bar(allTable.index, allTable, color = 'royalblue')
    ax1.set_xlabel(pred)
    ax1.set_ylabel('Number of Observations')
    ax1.set_xticks(allTable.index)
    ax1.grid(axis = 'y', linestyle = '--', linewidth = 0.5)

    ax2.plot(oddsTable.index, oddsTable, color = 'green', marker = 'o')
    ax2.set_xlabel(pred)
    ax2.set_ylabel('Odds')
    ax2.grid(axis = 'y', linestyle = '--', linewidth = 0.5)

    plt.show()

# Explore the continuous predictor
if (len(intName) > 0):
    print('=== Descriptive Statistics of Continuous Predictors ===')
    print(trainData[intName].describe())

for pred in intName:

    # Generate the contingency table of the interval input feature by the target
    cntTable = pandas.crosstab(index = trainData[pred], columns = trainData[yName], margins = False, dropna = True)

    # Calculate the row percents
    pctTable = 100.0 * cntTable.div(cntTable.sum(1), axis = 'index')
    yCat = cntTable.columns

    fig, ax = plt.subplots(dpi = 200, figsize = (10,4))
    plt.stackplot(pctTable.index, numpy.transpose(pctTable), baseline = 'zero', colors = cmap, labels = yCat)
    ax.xaxis.set_major_locator(MultipleLocator(base = 1000))
    ax.xaxis.set_minor_locator(MultipleLocator(base = 200))
    ax.yaxis.set_major_locator(MultipleLocator(base = 20))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f%%'))
    ax.yaxis.set_minor_locator(MultipleLocator(base = 5))
    ax.set_xlabel(pred)
    ax.set_ylabel('Percent')
    plt.grid(axis = 'both')
    plt.legend(loc = 'lower center', bbox_to_anchor = (0.5, 1), ncol = 3)
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
stepSummary = pandas.DataFrame()

# Intercept only model
resultList = Regression.BLogisticModel (X0_train, y_train, offset = None, maxIter = maxIter, tolSweep = tolS)

llk0 = resultList[3]
df0 = len(resultList[4])
stepSummary = pandas.concat([stepSummary, pandas.DataFrame([['Intercept', ' ', df0, llk0, numpy.NaN, numpy.NaN, numpy.NaN]])],
                            axis = 0, ignore_index = True)
stepSummary.columns = ['Predictor', 'Type', 'ModelDF', 'ModelLLK', 'DevChiSq', 'DevDF', 'DevSig']

print('======= Step Detail =======')
print('Step = ', 0)
print('Step Statistics:')
print(stepSummary)

cName = catName.copy()
iName = intName.copy()
entryThreshold = 0.05

for step in range(nPredictor):
    enterName = ''
    stepDetail = pandas.DataFrame()

    # Enter the next predictor
    for X_name in cName:
        X_train = pandas.get_dummies(trainData[[X_name]])
        X_train = X0_train.join(X_train)
        resultList = Regression.BLogisticModel (X_train, y_train, offset = None, maxIter = maxIter, tolSweep = tolS)
        llk1 = resultList[3]
        df1 = len(resultList[4])
        devChiSq = 2.0 * (llk1 - llk0)
        devDF = df1 - df0
        devSig = chi2.sf(devChiSq, devDF)
        stepDetail = pandas.concat([stepDetail, pandas.DataFrame([[X_name, 'categorical', df1, llk1, devChiSq, devDF, devSig]])],
                                   axis = 0, ignore_index = True)

    for X_name in iName:
        X_train = trainData[[X_name]]
        X_train = X0_train.join(X_train)
        resultList = Regression.BLogisticModel (X_train, y_train, offset = None, maxIter = maxIter, tolSweep = tolS)
        llk1 = resultList[3]
        df1 = len(resultList[4])
        devChiSq = 2.0 * (llk1 - llk0)
        devDF = df1 - df0
        devSig = chi2.sf(devChiSq, devDF)
        stepDetail = pandas.concat([stepDetail, pandas.DataFrame([[X_name, 'interval', df1, llk1, devChiSq, devDF, devSig]])],
                                   axis = 0, ignore_index = True)

    stepDetail.columns = ['Predictor', 'Type', 'ModelDF', 'ModelLLK', 'DevChiSq', 'DevDF', 'DevSig']

    # Find a predictor to enter, if any
    stepDetail.sort_values(by = 'DevSig', axis = 0, ascending = True, inplace = True)
    enterRow = stepDetail.iloc[0].copy()
    minPValue = enterRow['DevSig']
    if (minPValue <= entryThreshold):
        stepSummary = pandas.concat([stepSummary, pandas.DataFrame([enterRow])], axis = 0, ignore_index = True)
        df0 = enterRow['ModelDF']
        llk0 = enterRow['ModelLLK']

        enterName = enterRow['Predictor']
        enterType = enterRow['Type']
        if (enterType == 'categorical'):
            X_train = pandas.get_dummies(trainData[[enterName]].astype('category'))
            X0_train = X0_train.join(X_train)
            cName.remove(enterName)
        elif (enterType == 'interval'):
            X_train = trainData[[enterName]]
            X0_train = X0_train.join(X_train)
            iName.remove(enterName)
    else:
        break

    # Print debugging output
    print('======= Step Detail =======')
    print('Step = ', step+1)
    print('Step Statistics:')
    print(stepDetail)
    print('Enter predictor = ', enterName)
    print('Minimum P-Value =', minPValue)
    print('\n')

# End of forward selection
print('======= Step Summary =======')
print(stepSummary)

# Final model
resultList = Regression.BLogisticModel (X0_train, y_train, offset = None, maxIter = maxIter, tolSweep = tolS)
