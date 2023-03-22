# -*- coding: utf-8 -*-
"""
@Name: Week 9 Simple Cohort Study.py
@Creation Date: March 7, 2022
@author: Ming-Long Lam, Ph.D.
@organization: University of Chicago
(C) All Rights Reserved.
"""

import matplotlib.pyplot as plt
import numpy
import pandas
import sys

from scipy.stats import norm

# Set some options for printing all the columns
numpy.set_printoptions(precision = 10, threshold = sys.maxsize)
numpy.set_printoptions(linewidth = numpy.inf)

pandas.set_option('display.max_columns', None)
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', None)

pandas.options.display.float_format = '{:,.7f}'.format

trainData = pandas.read_csv('SimpleCohortStudy.csv')

trainData['Survival Time'] = numpy.where(numpy.isnan(trainData['Year of Death']), trainData['Year of Last Contact'], trainData['Year of Death'])
trainData['Status'] = numpy.where(numpy.isnan(trainData['Year of Death']), 'Censored', 'Death')

# Number of observation units at risk
nUnit = trainData.shape[0]

# Distribution of Status
statusDistribution = trainData.groupby('Status').size()
plt.figure(dpi = 200)
plt.bar(statusDistribution.index, statusDistribution, color = 'royalblue')
plt.xlabel('Survival Status')
plt.ylabel('Number of Respondents')
plt.yticks(range(0,16,2))
plt.grid(axis = 'y')
plt.show()

# Calculate the Kaplan-Meier Product Limit Estimator for the Survival Function
xtab = pandas.crosstab(index = trainData['Survival Time'], columns = trainData['Status'])

lifeTable = pandas.DataFrame({'Survival Time': 0, 'Number Left': nUnit, 'Number of Events': 0, 'Number Censored': 0}, index = [0])
lifeTable = lifeTable.append(pandas.DataFrame({'Survival Time': xtab.index, 'Number of Events': xtab['Death'].to_numpy(),
                                               'Number Censored': xtab['Censored'].to_numpy()}),
                             ignore_index = True)

lifeTable['Number at Risk'] = nUnit

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
plt.plot(lifeTable['Survival Time'], lifeTable['Prob Survival'], marker = '+', markersize = 10, drawstyle = 'steps')
plt.plot(lifeTable['Survival Time'], lifeTable['Upper CI Prob Survival'], marker = '+',
         markersize = 10, drawstyle = 'steps', linestyle = 'dashed', label = 'Upper Confidence Limit')
plt.plot(lifeTable['Survival Time'], lifeTable['Lower CI Prob Survival'], marker = '+',
         markersize = 10, drawstyle = 'steps', linestyle = 'dashed', label = 'Lower Confidence Limit')
plt.xlabel('Surivial Time (Years)')
plt.ylabel('Survival Function')
plt.xticks(numpy.arange(0,26,2))
plt.yticks(numpy.arange(0.0,1.1,0.1))
plt.grid(axis = 'both')
plt.legend()
plt.show()

plt.figure(dpi = 200)
plt.plot(lifeTable['Survival Time'], lifeTable['Prob Survival'], marker = '+', markersize = 10, drawstyle = 'steps')
plt.fill_between(lifeTable['Survival Time'], lifeTable['Lower CI Prob Survival'], lifeTable['Upper CI Prob Survival'],
                 color = 'lightgreen', label = '95% Confidence Band')
plt.xlabel('Surivial Time (Years)')
plt.ylabel('Survival Function')
plt.xticks(numpy.arange(0,26,2))
plt.yticks(numpy.arange(0.0,1.1,0.1))
plt.grid(axis = 'both')
plt.legend()
plt.show()

plt.figure(dpi = 200)
plt.plot(lifeTable['Survival Time'], lifeTable['Cumulative Hazard'], marker = '+', markersize = 10, drawstyle = 'steps')
plt.xlabel('Survivial Time (Years)')
plt.ylabel('Cumulative Hazard Function')
plt.xticks(numpy.arange(0,26,2))
plt.yticks(numpy.arange(0.0,0.8,0.1))
plt.grid(axis = 'both')
plt.show()
