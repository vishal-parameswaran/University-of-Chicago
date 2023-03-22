import pandas as pd

# -*- coding: utf-8 -*-
"""
@name: Week 3 Poisson GLM.py
@author: Ming-Long Lam, Ph.D.
@organization: University of Chicago
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas
import sys
# Set some options for printing all the columns
np.set_printoptions(precision = 10, threshold = sys.maxsize)
np.set_printoptions(linewidth = np.inf)
pandas.set_option('display.max_columns', None)
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', None)
pandas.options.display.float_format = '{:,.10}'.format
import seaborn as sns
import math
from scipy.stats import chi2
from scipy.stats import norm

import Regression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

target = 'CLM_AMT'
exposure = 'EXPOSURE'
cat_cols = ["CAR_TYPE", "CAR_USE", "EDUCATION", "GENDER", "MSTATUS", "PARENT1","RED_CAR","REVOKED","URBANICITY"]
int_pred = ["CAR_TYPE", "CAR_USE", "EDUCATION", "GENDER", "MSTATUS", "PARENT1", "RED_CAR", "REVOKED", "URBANICITY",'HOMEKIDS', 'KIDSDRIV', "AGE", "BLUEBOOK", "CAR_AGE", "HOME_VAL", "INCOME", "YOJ", "MVR_PTS", "TIF","TRAVTIME"]
claim_history = pd.read_excel('claim_history.xlsx')
claim_history[["BLUEBOOK", "HOME_VAL", "INCOME"]] = claim_history[["BLUEBOOK", "HOME_VAL", "INCOME"]] / 1000

train_data = claim_history[[target, exposure] + int_pred]
train_data = train_data[train_data[exposure] > 0.0].dropna().reset_index(drop = True)
y_train = train_data[target]
o_train = np.log(train_data[exposure])
train_data.head()

# Estimate the Tweedie's P value
xtab = pd.pivot_table(train_data, values = target, index = cat_cols,columns = None, aggfunc = ['count', 'mean', 'var'])
cell_stats = xtab[['mean','var']].reset_index().droplevel(1, axis = 1)
ln_Mean = np.where(cell_stats['mean'] > 1e-16, np.log(cell_stats['mean']), np.NaN)
ln_Variance = np.where(cell_stats['var'] > 1e-16, np.log(cell_stats['var']), np.NaN)
use_cell = np.logical_not(np.logical_or(np.isnan(ln_Mean), np.isnan(ln_Variance)))
X_train = ln_Mean[use_cell]
y_train = ln_Variance[use_cell]
# Scatterplot of lnVariance vs lnMean
plt.figure(figsize = (6,6))
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
tweediePhi = np.exp(result_list[0][0])
print("################################################################################################################")
print("Answer 1.a)")
print("Tweedie Power: " + str(tweediePower) + " Tweedie Phi: " + str(tweediePhi))

def create_categorical_value(dfs, column_name):
    u = dfs[column_name].astype('category')
    u_freq = u.value_counts(ascending=True)
    pm = u.cat.reorder_categories(list(u_freq.index))
    term_df_type = pandas.get_dummies(pm)
    term_df_type = term_df_type.add_suffix("_"+column_name)
    return term_df_type
def run_against_all(dfs, init_cols, categorical_cols, previous_train, output_train,offset_train, qll, dv):
    step_detail = []
    output_list = []
    for i in init_cols:
        if i in categorical_cols:
            new_train = previous_train.join(create_categorical_value(dfs, i), rsuffix=i)
        else:
            new_train = previous_train.join(dfs[i], rsuffix=i)
        regression_output = Regression.TweedieRegression(new_train, output_train,offset = offset_train, tweedieP = tweediePower)
        qll_1 = regression_output[3]
        dv_1 = len(regression_output[4])
        phi = regression_output[7]
        dev_chisq = 2 * (qll_1 - qll) / phi
        deviance_df = dv_1 - dv
        dev_sig = chi2.sf(dev_chisq, deviance_df)
        step_detail.append([str(i), dv_1, qll_1,dev_chisq, deviance_df, dev_sig])
        output_list.append(regression_output)
    step_df = pd.DataFrame(step_detail)
    chosen_value = step_df[step_df[5] == step_df[5].min()]
    chosen_output = output_list[step_df.index[step_df[5] == step_df[5].min()].tolist()[0]]
    return chosen_value.values, chosen_output
def forward_selector(data, target_col, init_cols, cat_columns,offset_col):
    # First Run
    target_train = data[target_col]
    intercept_train = data[[target_col]]
    init = init_cols
    intercept_train.insert(0, 'Intercept', 1.0)
    intercept_train = intercept_train.drop(columns=target_col)
    offset_train = np.log(data[offset_col])
    intercept_result = Regression.TweedieRegression(intercept_train, target_train, offset = offset_train, tweedieP=tweediePower)
    step_summary = []
    qll = intercept_result[3]
    dv = len(intercept_result[4])
    dev_sig = 0
    threshold = 0.05
    value_outputs = None
    step_summary.append(['Intercept', dv, qll, np.nan, np.nan, np.nan])
    while dev_sig < threshold:
        run, s = run_against_all(data, init, cat_columns, intercept_train, target_train,offset_train, qll, dv)
        run = run[0]
        qll = run[2]
        dv = run[1]
        dev_sig = run[5]
        if dev_sig < threshold:
            value_outputs = s
            if run[0] in cat_columns:
                intercept_train = intercept_train.join(create_categorical_value(data, run[0]), rsuffix=str(run[0]))
            else:
                intercept_train = intercept_train.join(data[run[0]], rsuffix=str(run[0]))
            init.remove(run[0])
            step_summary.append([run[0], dv, qll, run[3], run[4], run[5]])
    step_summary = pd.DataFrame(step_summary,columns=  ["Predictor","Model Degree of Freedom","Quasi Log Likelihood","Deviance Chi Square","Degree of Freedom","Chi-square significance"])
    step_summary.index.rename("Step Number")
    return step_summary, value_outputs, intercept_result
int_pred = ["CAR_TYPE", "CAR_USE", "EDUCATION", "GENDER", "MSTATUS", "PARENT1", "RED_CAR", "REVOKED", "URBANICITY",
            'HOMEKIDS', 'KIDSDRIV', "AGE", "BLUEBOOK", "CAR_AGE", "HOME_VAL", "INCOME", "YOJ", "MVR_PTS", "TIF",
            "TRAVTIME"]
df, outps, train = forward_selector(train_data, target, int_pred, cat_cols,exposure)
print("###############################################################################################################")
print("Answer 1.b)")
print(df)


def distance_corr(x, y):
    # Calculate the Adjusted Distance for x
    x_new = pd.DataFrame(x)
    y_new = pd.DataFrame(y)
    x_new = x_new.dropna()
    y_new = y_new.dropna()
    x_new = x_new.fillna(0)
    i = x_new.to_numpy().T[0]
    n = len(i)
    distance_matrix_x = pd.DataFrame(np.abs(i[:, None] - i))
    distance_matrix_x = distance_matrix_x.sub(distance_matrix_x.mean(axis=0), axis=1)
    distance_matrix_x = distance_matrix_x.sub(distance_matrix_x.mean(axis=1), axis=0)
    emp_dist_x = distance_matrix_x.pow(2)
    emp_dist_x = emp_dist_x.values.sum() / (n ** 2)

    # Calculate the Adjusted Distance for y
    y_new = y_new.fillna(0)
    d = y_new.to_numpy().T[0]
    n = len(i)
    distance_matrix_y = pd.DataFrame(np.abs(d[:, None] - d))
    distance_matrix_y = distance_matrix_y.sub(distance_matrix_y.mean(axis=0), axis=1)
    distance_matrix_y = distance_matrix_y.sub(distance_matrix_y.mean(axis=1), axis=0)
    emp_dist_y = distance_matrix_y.pow(2)
    emp_dist_y = emp_dist_y.values.sum() / (n ** 2)

    # Calculate the Distance covariance
    emp_dist_xy = distance_matrix_x.mul(distance_matrix_y, axis="index")
    emp_dist_xy = emp_dist_xy.values.sum() / (n ** 2)

    # Calculate the Distance Correlation Coefficient

    if (emp_dist_x * emp_dist_y) != 0:
        distance_correlation_coeff = math.sqrt(emp_dist_xy / (math.sqrt(emp_dist_x * emp_dist_y)))
    else:
        distance_correlation_coeff = np.nan
    return distance_correlation_coeff

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
   
   dev_x = x - np.mean(x)
   dev_y = y - np.mean(y)
   
   ss_xx = np.mean(dev_x * dev_x)
   ss_yy = np.mean(dev_y * dev_y)

   if (ss_xx > 0.0 and ss_yy > 0.0):
      ss_xy = np.mean(dev_x * dev_y)
      rho = (ss_xy / ss_xx) * (ss_xy / ss_yy)
      rho = np.sign(ss_xy) * np.sqrt(rho)
   else:
      rho = np.nan
      
   return (rho)
def find_metrics(y_actual, y_predicted):
      y_simple_residual = y_actual - y_predicted
      mse = np.mean(np.power(y_simple_residual, 2))
      rmse = np.sqrt(mse)
      rel_error = mse / np.var(y_actual, ddof = 0)
      dist_corr = distance_corr(y_actual, y_predicted)
      pearson_corr = PearsonCorrelation(y_actual, y_predicted)
      output_metric = pd.DataFrame({"RMSE": [np.round(rmse, 8)], "Relative Error": [rel_error],"Pearson Correlation": [np.round(pearson_corr, 8)],"Distance Correlation": [np.round(dist_corr, 8)]})
      return output_metric
y_train = train_data[target].values
y_pred = outps[6]
print("###############################################################################################################")
print("Answer 1.c)")
print(find_metrics(y_train, outps[6]))

print("###############################################################################################################")
print("Answer 1.d)")
print(outps[0])
print("Dispersion Scale Parameter: " + str(outps[7]))

y_pred_A = train[6].reindex(train_data.index)
y_pred_A.name = 'Intercept Only Model'

y_pred_B = outps[6].reindex(train_data.index)
y_pred_B.name = 'Final Model'

prediction = claim_history[[target, exposure]].join(y_pred_A).join(y_pred_B).dropna()
column_sums = np.sum(prediction[['EXPOSURE', 'CLM_AMT','Intercept Only Model','Final Model']], axis = 0)

adjP_CLM_AMT_A = prediction['Intercept Only Model'] * (column_sums['CLM_AMT'] / column_sums['Intercept Only Model'])
adjP_CLM_AMT_B = prediction['Final Model'] * (column_sums['CLM_AMT'] / column_sums['Final Model'])

prediction = prediction.join(pandas.DataFrame({'AdjIntercept Only Model': adjP_CLM_AMT_A, 'AdjFinal Model': adjP_CLM_AMT_B}))
prediction['impact'] = adjP_CLM_AMT_B / adjP_CLM_AMT_A

prediction.sort_values(by = 'impact', axis = 0, ascending = True, inplace = True)
prediction['Cumulative Exposure'] = prediction['EXPOSURE'].cumsum()

cumulative_exposure_cutoff = np.arange(0.1, 1.1, 0.1) * column_sums['EXPOSURE']
decile = np.zeros_like(prediction['Cumulative Exposure'], dtype = int)
for i in range(10):
   decile = decile + np.where(prediction['Cumulative Exposure'] > cumulative_exposure_cutoff[i], 1, 0)
   
prediction['decile'] = decile + 1

xtab = pandas.pivot_table(prediction, index = 'decile', columns = None,
                          values = ['EXPOSURE','CLM_AMT','AdjIntercept Only Model', 'AdjFinal Model'],
                          aggfunc = ['sum'])

loss_ratio_A = xtab['sum','CLM_AMT'] / xtab['sum','AdjIntercept Only Model']
loss_ratio_B = xtab['sum','CLM_AMT'] / xtab['sum','AdjFinal Model']

MAE_A = np.mean(np.abs((loss_ratio_A - 1.0)))
MAE_B = np.mean(np.abs((loss_ratio_B - 1.0)))
print("############################################################################")
print("Answer 1.e)")
plt.figure(figsize = (10,6))
plt.plot(xtab.index, loss_ratio_A, marker = 'o', label = 'Intercept Only Model')
plt.plot(xtab.index, loss_ratio_B, marker = 'X', label = 'Final Model')
plt.xlabel('Decile of Impact (= Adj. Predicted Loss B / Adj. Predicted Loss A)')
plt.ylabel('Actual Loss / Adj. Predicted Loss')
plt.xticks(range(1,11))
plt.grid()
plt.legend()
plt.show()

# Question 2
print("############################################################################")
print("Question 2)")

myeloma_data = pd.read_csv('myeloma.csv')

nUnit = myeloma_data.shape[0]
xtab = pandas.crosstab(index = myeloma_data['Time'], columns = myeloma_data['VStatus'])
lifeTable = pandas.DataFrame({'Time': 0, 'Number Left': nUnit, 'Number of Events': 0, 'Number Censored': 0}, index = [0])
lifeTable = lifeTable.append(pandas.DataFrame({'Time': xtab.index, 'Number of Events': xtab[1].to_numpy(),
                                               'Number Censored': xtab[0].to_numpy()}),
                             ignore_index = True)

lifeTable['Number at Risk'] = nUnit

nTime = lifeTable.shape[0]
probSurvival = 1.0
hazardFunction = 0.0
seProbSurvival = 0.0
lifeTable.at[0,'Prob Survival'] = probSurvival
lifeTable.at[0,'Prob Failure'] = 1.0 - probSurvival
lifeTable.at[0,'Cumulative Hazard'] = hazardFunction

for i in np.arange(1,nTime):
   nDeath = lifeTable.at[i,'Number of Events']
   nAtRisk = lifeTable.at[i-1,'Number Left'] - lifeTable.at[i-1,'Number Censored']
   nLeft = nAtRisk - nDeath
   probSurvival = probSurvival * (nLeft / nAtRisk)
   if nLeft != 0:
      seProbSurvival = seProbSurvival + nDeath / nAtRisk / nLeft
   else:
      seProbSurvival = np.nan
   hazardFunction = hazardFunction + (nDeath / nAtRisk)
   lifeTable.at[i, 'SE Prob Survival'] = seProbSurvival
   lifeTable.at[i,'Number Left'] = nLeft
   lifeTable.at[i,'Number at Risk'] = nAtRisk
   lifeTable.at[i,'Prob Survival'] = probSurvival
   lifeTable.at[i,'Prob Failure'] = 1.0 - probSurvival
   lifeTable.at[i,'Cumulative Hazard'] = hazardFunction

lifeTable['SE Prob Survival'] = lifeTable['Prob Survival'] * np.sqrt(lifeTable['SE Prob Survival'])

CIHalfWidth = norm.ppf(0.975) * lifeTable['SE Prob Survival']
u = lifeTable['Prob Survival'] - CIHalfWidth
lifeTable['Lower CI Prob Survival'] = np.where(u < 0.0, 0.0, u)

u = lifeTable['Prob Survival'] + CIHalfWidth
print("############################################################################")
print("Answer 2.a)")
print("Number of Risk Sets = "lifeTable.shape[0]-1)

lifeTable['Upper CI Prob Survival'] = np.where(u > 1.0, 1.0, u)
print("############################################################################")
print("Answer 2.b)")
print(lifeTable)

print("############################################################################")
print("Answer 2.c)")
print("Probability Survival: " + str(lifeTable[lifeTable.Time == 18]['Prob Survival'].values[0]) + " Cumulative Hazard: " + str(lifeTable[lifeTable.Time == 18]['Cumulative Hazard'].values[0]))

print("############################################################################")
print("Answer 2.d)")

plt.plot(lifeTable['Time'], lifeTable['Prob Survival'], marker = '+', markersize = 10, drawstyle = 'steps',label = 'Probability of Survival')
plt.fill_between(lifeTable['Time'], lifeTable['Lower CI Prob Survival'], lifeTable['Upper CI Prob Survival'],
                 color = 'lightgreen', label = '95% Confidence Band')
plt.xlabel('Surivial Time (Years)')
plt.ylabel('Survival Function')
plt.xticks(np.arange(0,lifeTable['Time'].max(),12))
plt.yticks(np.arange(0.0,1.1,0.1))
plt.grid(axis = 'both')
plt.legend()
plt.show()

print("############################################################################")
print("Answer 2.e)")
ts = lifeTable[lifeTable['Prob Survival'] == lifeTable[lifeTable['Prob Survival']>=0.5]['Prob Survival'].min()]['Time'].values[0]
ps = lifeTable[lifeTable['Prob Survival']>=0.5]['Prob Survival'].min()
tr = lifeTable[lifeTable['Prob Survival'] == lifeTable[lifeTable['Prob Survival']<0.5]['Prob Survival'].max()]['Time'].values[0]
pr = lifeTable[lifeTable['Prob Survival']<0.5]['Prob Survival'].max()
median_suvival_time = ts + ((ps-0.5)/(ps-pr))*(tr-ts)
print("Median Survival Time: " + str(round(median_suvival_time,10)))