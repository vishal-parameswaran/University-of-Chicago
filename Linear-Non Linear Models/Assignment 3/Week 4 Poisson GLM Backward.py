# -*- coding: utf-8 -*-
"""
@name: Week 4 Poisson GLM Backward.py
@author: Ming-Long Lam, Ph.D.
@organization: University of Chicago
"""
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

claim_history = pandas.read_excel('C:\\MScAnalytics\\Data\\claim_history.xlsx')

trainData = claim_history[['CLM_COUNT', 'EXPOSURE', 'CAR_TYPE', 'HOMEKIDS', 'KIDSDRIV']].dropna()
trainData.reset_index(inplace = True)

# Reorder the categories in ascending order of frequencies of the target field
u = trainData['CAR_TYPE'].astype('category')
u_freq = u.value_counts(ascending = True)
pm = u.cat.reorder_categories(list(u_freq.index))
term_car_type = pandas.get_dummies(pm)

term_homekids = trainData[['HOMEKIDS']]
term_kidsdriv = trainData[['KIDSDRIV']]

y_train = trainData['CLM_COUNT']
o_train = numpy.log(trainData['EXPOSURE'])

# Intercept + CAR_TYPE + HOMEKIDS + KIDSDRIV model
X_train = term_car_type.join(term_homekids)
X_train = X_train.join(term_kidsdriv)
X_train.insert(0, 'Intercept', 1.0)

step_summary = []

outList = Regression.PoissonRegression(X_train, y_train, o_train)
llk_0 = outList[3]
df_0 = len(outList[4])
step_summary.append(['_ALL_', df_0, llk_0, numpy.nan, numpy.nan, numpy.nan])

# Remove the first predictor
step_detail = []

# Try Removing CAR_TYPE
X_train = term_homekids.join(term_kidsdriv)
X_train.insert(0, 'Intercept', 1.0)

outList = Regression.PoissonRegression(X_train, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_0 - llk_1)
deviance_df = df_0 - df_1
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail.append(['- CAR_TYPE', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig])

# Try Removing HOMEKIDS
X_train = term_car_type.join(term_kidsdriv)
X_train.insert(0, 'Intercept', 1.0)

outList = Regression.PoissonRegression(X_train, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_0 - llk_1)
deviance_df = df_0 - df_1
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail.append(['- HOMEKIDS', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig])

# Try Removing KIDSDRIV
X_train = term_car_type.join(term_homekids)
X_train.insert(0, 'Intercept', 1.0)

outList = Regression.PoissonRegression(X_train, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_0 - llk_1)
deviance_df = df_0 - df_1
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail.append(['- KIDSDRIV', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig])
