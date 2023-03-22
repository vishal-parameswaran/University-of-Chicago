# -*- coding: utf-8 -*-
"""
@name: Week 4 Poisson GLM Forward.py
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

# Intercept only model
X_train = trainData[['CLM_COUNT']].copy()
X_train.insert(0, 'Intercept', 1.0)
X_train.drop(columns = ['CLM_COUNT'], inplace = True)

step_summary = []

outList = Regression.PoissonRegression(X_train, y_train, o_train)
llk_0 = outList[3]
df_0 = len(outList[4])
step_summary.append(['Intercept', df_0, llk_0, numpy.nan, numpy.nan, numpy.nan])

# Find the first predictor
step_detail = []

# Try Intercept + CAR_TYPE
X = X_train.join(term_car_type)
outList = Regression.PoissonRegression(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail.append(['+ CAR_TYPE', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig])

# Try Intercept + HOMEKIDS
X = X_train.join(term_homekids)
outList = Regression.PoissonRegression(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail.append(['+ HOMEKIDS', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig])

# Try Intercept + KIDSDRIV
X = X_train.join(term_kidsdriv)
outList = Regression.PoissonRegression(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail.append(['+ KIDSDRIV', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig])

# Current model is Intercept + CAR_TYPE
row = step_detail[step_detail[0] == '+ CAR_TYPE']
llk_0 = row.iloc[0][2]
df_0 = row.iloc[0][1]
step_summary = step_summary.append(row, ignore_index = True)
X_train = X_train.join(term_car_type)

# Find the second predictor
step_detail = pandas.DataFrame()

# Try Intercept + CAR_TYPE + HOMEKIDS
X = X_train.join(term_homekids)
outList = Regression.PoissonRegression(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail.append(['+ HOMEKIDS', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig])

# Try Intercept + CAR_TYPE + KIDSDRIV
X = X_train.join(term_kidsdriv)
outList = Regression.PoissonRegression(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail.append(['+ KIDSDRIV', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig])

# Current model is Intercept + CAR_TYPE + HOMEKIDS
row = step_detail[step_detail[0] == '+ HOMEKIDS']
llk_0 = row.iloc[0][2]
df_0 = row.iloc[0][1]
step_summary = step_summary.append(row, ignore_index = True)
X_train = X_train.join(term_homekids)

# Find the third predictor
step_detail = pandas.DataFrame()

# Try Intercept + CAR_TYPE + HOMEKIDS + KIDSDRIV
X = X_train.join(term_kidsdriv)
outList = Regression.PoissonRegression(X, y_train, o_train)
llk_1 = outList[3]
df_1 = len(outList[4])

deviance_chisq = 2 * (llk_1 - llk_0)
deviance_df = df_1 - df_0
deviance_sig = chi2.sf(deviance_chisq, deviance_df)
step_detail.append(['+ KIDSDRIV', df_1, llk_1, deviance_chisq, deviance_df, deviance_sig])

# Current model is Intercept + CAR_TYPE + HOMEKIDS + KIDSDRIV
row = step_detail[step_detail[0] == '+ KIDSDRIV']
llk_0 = row[0][2]
df_0 = row[0][1]
step_summary.append(row)
X_train = X_train.join(term_kidsdriv)
