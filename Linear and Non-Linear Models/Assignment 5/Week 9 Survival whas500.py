# -*- coding: utf-8 -*-
"""
@Name: Week 9 Survival whas500.py
@Creation Date: March 7, 2022
@author: Ming-Long Lam, Ph.D.
@organization: University of Chicago
(C) All Rights Reserved.
"""

import matplotlib.pyplot as plt
import numpy
import pandas
import sys

from lifelines import CoxPHFitter
from scipy.stats import norm

sys.path.append('C:\\MScAnalytics\\Linear and Nonlinear Model\\Job')
import Regression

# Set some options for printing all the columns
numpy.set_printoptions(precision = 10, threshold = sys.maxsize)
numpy.set_printoptions(linewidth = numpy.inf)

pandas.set_option('display.max_columns', None)
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', None)

pandas.options.display.float_format = '{:,.7f}'.format

whas500 = pandas.read_csv('C:\\MScAnalytics\\Data\\whas500.csv')

nUnit = 500

# Distribution of Status
plt.figure(dpi = 200)
statusDistribution = whas500.groupby('fstat').size()
plt.bar(statusDistribution.index, statusDistribution, color = 'royalblue')
plt.xlabel('Survival Status')
plt.ylabel('Number of Respondents')
plt.xticks(range(2))
plt.yticks(range(0,320,20))
plt.grid(axis = 'y')
plt.show()

# Calculate the Kaplan-Meier Product Limit Estimator for the Survival Function
xtab = pandas.crosstab(index = whas500['lenfol'], columns = whas500['fstat'])

lifeTable = pandas.DataFrame({'Survival Time': 0, 'Number Left': nUnit, 'Number of Events': 0, 'Number Censored': 0}, index = [0])
lifeTable = pandas.concat([lifeTable, pandas.DataFrame({'Survival Time': xtab.index, 'Number of Events': xtab[1].to_numpy(),
                                                        'Number Censored': xtab[0].to_numpy()})],
                          axis = 0, ignore_index = True)

lifeTable[['Number at Risk']] = nUnit

nTime = lifeTable.shape[0]
probSurvival = 1.0
hazardFunction = 0.0
seProbSurvival = 0.0
lifeTable.at[0,'Prob Survival'] = probSurvival
lifeTable.at[0,'Prob Failure'] = 1.0 - probSurvival
lifeTable.at[0,'Cumulative Hazard'] = hazardFunction

for i in numpy.arange(1,nTime):
   nDeath = lifeTable.at[i,'Number of Events']
   nAtRisk = lifeTable.at[i-1,'Number Left'] - lifeTable.at[i-1,'Number Censored']
   nLeft = nAtRisk - nDeath
   probSurvival = probSurvival * (nLeft / nAtRisk)
   seProbSurvival = seProbSurvival + nDeath / nAtRisk / nLeft
   hazardFunction = hazardFunction + (nDeath / nAtRisk)
   lifeTable.at[i, 'SE Prob Survival'] = seProbSurvival
   lifeTable.at[i,'Number Left'] = nLeft
   lifeTable.at[i,'Number at Risk'] = nAtRisk
   lifeTable.at[i,'Prob Survival'] = probSurvival
   lifeTable.at[i,'Prob Failure'] = 1.0 - probSurvival
   lifeTable.at[i,'Cumulative Hazard'] = hazardFunction

lifeTable['SE Prob Survival'] = lifeTable['Prob Survival'] * numpy.sqrt(lifeTable['SE Prob Survival'])

CIHalfWidth = norm.ppf(0.975) * lifeTable['SE Prob Survival']
u = lifeTable['Prob Survival'] - CIHalfWidth
lifeTable['Lower CI Prob Survival'] = numpy.where(u < 0.0, 0.0, u)

u = lifeTable['Prob Survival'] + CIHalfWidth
lifeTable['Upper CI Prob Survival'] = numpy.where(u > 1.0, 1.0, u)

plt.figure(dpi = 200)
plt.plot(lifeTable['Survival Time'], lifeTable['Prob Survival'], drawstyle = 'steps')
plt.plot(lifeTable['Survival Time'], lifeTable['Upper CI Prob Survival'], drawstyle = 'steps',
         linestyle = 'dashed', label = 'Upper Confidence Limit')
plt.plot(lifeTable['Survival Time'], lifeTable['Lower CI Prob Survival'], drawstyle = 'steps',
         linestyle = 'dashed', label = 'Lower Confidence Limit')
plt.xlabel('Total Length of Follow-up\n(Days from Hospital Admission to Date of Last Follow-up)')
plt.ylabel('Survival Function')
plt.xticks(numpy.arange(0,2920,365))
plt.yticks(numpy.arange(0.0,1.1,0.1))
plt.grid(axis = 'both')
plt.legend()
plt.show()

# Plot the Survival Function with a Confidence Band
plt.plot(lifeTable['Survival Time'], lifeTable['Prob Survival'], drawstyle = 'steps')
plt.fill_between(lifeTable['Survival Time'], lifeTable['Lower CI Prob Survival'], lifeTable['Upper CI Prob Survival'],
                 color = 'lightgreen', label = '95% Confidence Band')
plt.xlabel('Total Length of Follow-up\n(Days from Hospital Admission to Date of Last Follow-up)')
plt.ylabel('Survival Function')
plt.xticks(numpy.arange(0,2920,365))
plt.yticks(numpy.arange(0.0,1.1,0.1))
plt.grid(axis = 'both')
plt.legend()
plt.show()

plt.figure(dpi = 200)
plt.plot(lifeTable['Survival Time'], lifeTable['Cumulative Hazard'], drawstyle = 'steps')
plt.xlabel('Total Length of Follow-up\n(Days from Hospital Admission to Date of Last Follow-up)')
plt.ylabel('Cumulative Hazard Function')
plt.xticks(numpy.arange(0,2920,365))
plt.yticks(numpy.arange(0.0,3.5,0.5))
plt.grid(axis = 'both')
plt.show()

aliveData = whas500[whas500['fstat'] == 0]
deadData = whas500[whas500['fstat'] == 1]

aliveData_MeanAge = numpy.mean(aliveData['age'])
deadData_MeanAge = numpy.mean(deadData['age'])

fig, (ax0, ax1) = plt.subplots(nrows = 1, ncols = 2, sharey = True, dpi = 200)
aliveData.boxplot('age', by = 'gender', ax = ax0)
ax0.axhline(y = aliveData_MeanAge, color = 'red', linestyle = '--', label = 'Mean')
ax0.set_title('Alive')
ax0.set_ylabel('Age at Hospital Admission')
deadData.boxplot('age', by = 'gender', ax = ax1)
ax1.axhline(y = deadData_MeanAge, color = 'red', linestyle = '--', label = 'Mean')
ax1.set_title('Dead')
plt.suptitle('')
plt.legend(loc = 'lower center', bbox_to_anchor=(-0.1, 1.01))
plt.show()

overall_MeanAge = numpy.mean(whas500['age'])
plt.figure(dpi = 200)
plt.scatter(aliveData['age'], aliveData['lenfol'], c = 'green', s = 20, label = 'Alive')
plt.scatter(deadData['age'], deadData['lenfol'], c = 'red', s = 20, label = 'Dead')
plt.axvline(x = overall_MeanAge, color = 'black', linestyle = '--', label = 'Mean')
plt.xlabel('Age at Hospital Admission')
plt.ylabel('Total Length of Follow-up')
plt.grid(axis = 'both')
plt.legend()
plt.show()

overall_MeanBMI = numpy.mean(whas500['bmi'])
plt.figure(dpi = 200)
plt.scatter(aliveData['bmi'], aliveData['lenfol'], c = 'green', s = 20, label = 'Alive')
plt.scatter(deadData['bmi'], deadData['lenfol'], c = 'red', s = 20, label = 'Dead')
plt.axvline(x = overall_MeanBMI, color = 'black', linestyle = '--', label = 'Mean')
plt.xlabel('Body Mass Index')
plt.ylabel('Total Length of Follow-up')
plt.grid(axis = 'both')
plt.legend()
plt.show()

overall_MeanSysbp = numpy.mean(whas500['sysbp'])
plt.figure(dpi = 200)
plt.scatter(aliveData['sysbp'], aliveData['lenfol'], c = 'green', s = 20, label = 'Alive')
plt.scatter(deadData['sysbp'], deadData['lenfol'], c = 'red', s = 20, label = 'Dead')
plt.axvline(x = overall_MeanSysbp, color = 'black', linestyle = '--', label = 'Mean')
plt.xlabel('Initial Systolic Blood Pressure mmHg')
plt.ylabel('Total Length of Follow-up')
plt.grid(axis = 'both')
plt.legend()
plt.show()

overall_MeanDiasbp = numpy.mean(whas500['diasbp'])
plt.figure(dpi = 200)
plt.scatter(aliveData['diasbp'], aliveData['lenfol'], c = 'green', s = 20, label = 'Alive')
plt.scatter(deadData['diasbp'], deadData['lenfol'], c = 'red', s = 20, label = 'Dead')
plt.axvline(x = overall_MeanDiasbp, color = 'black', linestyle = '--', label = 'Mean')
plt.xlabel('Initial Diastolic Blood Pressure mmHg')
plt.ylabel('Total Length of Follow-up')
plt.grid(axis = 'both')
plt.legend()
plt.show()

# gender + age
fullX = pandas.get_dummies(whas500[['gender']].astype('category'))
fullX = fullX.join(whas500[['age']])
fullX.insert(0, '_Intercept', 1.0)
XtX = numpy.transpose(fullX).dot(fullX)             # The SSCP matrix
pDim = XtX.shape[0]
origDiag = numpy.diag(XtX)
invXtX, aliasParam, nonAliasParam = Regression.SWEEPOperator(pDim, XtX, origDiag, sweepCol = range(pDim), tol = 1e-7)

print('Aliased Parameters:\n', fullX.columns[list(aliasParam)])

modelX = fullX.iloc[:, list(nonAliasParam)].drop('_Intercept', axis = 1)
modelX = modelX.join(whas500[['lenfol','fstat']])

cph = CoxPHFitter()
cph.fit(modelX, duration_col='lenfol', event_col='fstat')
print(cph.params_)
cph.print_summary()

baseHazard = cph.baseline_hazard_

plt.figure(dpi = 200)
plt.title('gender + age')
plt.plot(baseHazard.index, baseHazard['baseline hazard'], drawstyle = 'steps', marker = '+')
plt.xlabel('Total Length of Follow-up\n(Days from Hospital Admission to Date of Last Follow-up)')
plt.ylabel('Baseline Hazard Function')
plt.xticks(numpy.arange(0,2920,365))
plt.yticks(numpy.arange(0.0,5.0,0.5))
plt.grid(axis = 'both')
plt.show()

cph.check_assumptions(modelX, p_value_threshold=0.05, show_plots=True)

# age + sysbp + diasbp + bmi
fullX = whas500[['age','sysbp','diasbp','bmi']]
# age_bmi = whas500['age']*whas500['bmi']
# age_bmi.name = 'age * bmi'
# fullX = fullX.join(age_bmi)
fullX.insert(0, '_Intercept', 1.0)
XtX = numpy.transpose(fullX).dot(fullX)             # The SSCP matrix
pDim = XtX.shape[0]
origDiag = numpy.diag(XtX)
invXtX, aliasParam, nonAliasParam = Regression.SWEEPOperator(pDim, XtX, origDiag, sweepCol = range(pDim), tol = 1e-7)

print('Aliased Parameters:\n', fullX.columns[list(aliasParam)])

modelX = fullX.iloc[:, list(nonAliasParam)].drop('_Intercept', axis = 1)
modelX = modelX.join(whas500[['lenfol','fstat']])

cph = CoxPHFitter()
cph.fit(modelX, duration_col='lenfol', event_col='fstat')
cph.print_summary()

# age + bmi
fullX = whas500[['age','bmi']]
# age_bmi = whas500['age']*whas500['bmi']
# age_bmi.name = 'age * bmi'
# fullX = fullX.join(age_bmi)
fullX.insert(0, '_Intercept', 1.0)
XtX = numpy.transpose(fullX).dot(fullX)             # The SSCP matrix
pDim = XtX.shape[0]
origDiag = numpy.diag(XtX)
invXtX, aliasParam, nonAliasParam = Regression.SWEEPOperator(pDim, XtX, origDiag, sweepCol = range(pDim), tol = 1e-7)

print('Aliased Parameters:\n', fullX.columns[list(aliasParam)])

modelX = fullX.iloc[:, list(nonAliasParam)].drop('_Intercept', axis = 1)
modelX = modelX.join(whas500[['lenfol','fstat']])

cph = CoxPHFitter()
cph.fit(modelX, duration_col='lenfol', event_col='fstat')
cph.print_summary()

# Plot the coefficients
plt.figure(figsize = (6,4), dpi = 200)
cph.plot()
plt.show()

plt.figure(dpi = 200)
cph.plot_partial_effects_on_outcome(covariates = 'age',
                                    values=[30, 40, 50, 60, 70, 80], cmap='coolwarm',
                                    figsize = (8,6))
plt.xlabel('Length pf Follow-up')
plt.ylabel('Survival Probability')
plt.xticks(numpy.arange(0,2920,365))
plt.grid(axis = 'both')
plt.show()

print('Concordance Index = ', cph.concordance_index_)

print('Parameter Estimates = ', cph.params_)
 
predCH = cph.predict_cumulative_hazard(modelX)
predSF = cph.predict_survival_function(modelX)

baseHazard = cph.baseline_hazard_

plt.figure(dpi = 200)
plt.title('age + bmi')
plt.plot(baseHazard.index, baseHazard['baseline hazard'], drawstyle = 'steps', marker = '+')
plt.xlabel('Total Length of Follow-up\n(Days from Hospital Admission to Date of Last Follow-up)')
plt.ylabel('Baseline Hazard Function')
plt.xticks(numpy.arange(0,2920,365))
plt.yticks(numpy.arange(0.0,5.0,0.5))
plt.grid(axis = 'both')
plt.show()

cph.check_assumptions(modelX, p_value_threshold=0.05, show_plots=True)
