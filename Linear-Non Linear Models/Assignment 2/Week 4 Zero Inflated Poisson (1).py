# -*- coding: utf-8 -*-
"""
@name: Week 4 Zero Inflated Poisson.py
@author: Ming-Long Lam, Ph.D.
@organization: University of Chicago
"""
import matplotlib.pyplot as plt
import numpy
import pandas
import sys

from scipy.special import gammaln
from scipy.stats import norm, chi2

import seaborn

sys.path.append('C:\\MScAnalytics\\Linear and Nonlinear Model\\Job')
import Regression

# Set some options for printing all the columns
numpy.set_printoptions(precision = 10, threshold = sys.maxsize)
numpy.set_printoptions(linewidth = numpy.inf)

pandas.set_option('display.max_columns', None)
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', None)

pandas.options.display.float_format = '{:,.7e}'.format

target = 'CLM_COUNT'
exposure = 'EXPOSURE'
cat_pred = ['CAR_TYPE']
int_pred = ['HOMEKIDS', 'KIDSDRIV']

claim_history = pandas.read_excel('C:\\MScAnalytics\\Data\\claim_history.xlsx')

train_data = claim_history[claim_history[exposure] > 0.0]            # Only positive exposure
train_data = train_data[[target] + [exposure] + cat_pred + int_pred] # Only necessary variables
train_data = train_data.dropna().reset_index()                       # Remove missing values

# Missing value situation
print(train_data.isnull().sum())

# Reorder the categories in ascending order of frequencies of the categorical field
for  pred in cat_pred:
   u = train_data[pred].astype('category')
   u_freq = u.value_counts(ascending = True)
   train_data[pred] = u.cat.reorder_categories(list(u_freq.index))

# Display relationship in heatmap
for pred in cat_pred + int_pred:
    xtab = pandas.crosstab(train_data[target], train_data[pred],
                           values = train_data[exposure], aggfunc = numpy.mean)
    xtab.fillna(0, inplace = True)
    print('\nPredictor: ', pred)
    print(xtab)

    plt.figure(figsize = (10,8), dpi = 200)
    ax = seaborn.heatmap(xtab, cmap = 'PiYG', cbar_kws = {'label': 'Mean Exposure'})
    ax.invert_yaxis()    
    plt.show()

# Claim Amount model is CLM_COUNT = Intercept + CAR_TYPE + HOMEKIDS + KIDSDRIV
X_train = pandas.get_dummies(train_data[cat_pred])
X_train = X_train.join(train_data[int_pred])

# Probability of filing claim model is Intercept + CAR_TYPE + HOMEKIDS + KIDSDRIV
U_train = pandas.get_dummies(train_data[cat_pred])
U_train = U_train.join(train_data[int_pred])

y_train = train_data[target]
n_sample = y_train.shape[0]

e_train = train_data[exposure]
o_train_X = numpy.log(e_train)
o_train_U = numpy.zeros(n_sample)

# Precompute the ln(y!)
constLLK = gammaln(y_train+1.0)

# Generate the model matrix with the Intercept term as the first column
modelX = X_train.copy()
modelX.insert(0, 'Intercept', 1.0)
modelXT = modelX.transpose()
param_name_X = modelX.columns
n_param_X = modelX.shape[1]

modelU = U_train.copy()
modelU.insert(0, 'Intercept', 1.0)
modelUT = modelU.transpose()
param_name_U = modelU.columns
n_param_U = modelU.shape[1]

n_param = n_param_X + n_param_U

# Specify some constants
maxIter = 20
maxStep = 5
tolSweep = 1e-7
tolLLK = 1e-3
tolParameter = 1e-10

# Initialize parameter array
beta = numpy.zeros(n_param_X)
beta[0] = numpy.log(numpy.mean(y_train))
nu_X = o_train_X + modelX.dot(beta)
lamda = numpy.exp(nu_X)

pi_0 = numpy.mean(numpy.where(y_train > 0.0, 0.0, 1.0))
gamma = numpy.zeros(n_param_U)
gamma[0] = numpy.log(pi_0 / (1.0 - pi_0))
nu_U = o_train_U + modelU.dot(gamma)
vu = numpy.exp(nu_U)

y_pred = lamda / (1.0 + vu)

llk = numpy.sum(numpy.where(y_train > 0.0, (y_train * nu_X - lamda - constLLK), (numpy.log(vu + numpy.exp(-lamda)))))
llk = llk - numpy.sum(numpy.log(1.0 + vu))

# Prepare the iteration history table
itList = [0, llk, 0]
itList.extend(beta)
itList.extend(gamma)
iterTable = [itList]

n_param = n_param_X + n_param_U
gradient = numpy.zeros(n_param)
hessian = numpy.zeros((n_param, n_param))

for it in range(maxIter):
   prob_pi = vu / (1.0 + vu)
   vel = vu * numpy.exp(lamda)
   vel2 = numpy.power((1.0 + vel), 2.0)

   vec_X = numpy.where(y_train > 0.0, (y_train - lamda), (-lamda / (1.0 + vel)))
   gradient[0:n_param_X] = modelXT.dot(vec_X)
   vec_U = numpy.where(y_train > 0.0, 0.0, (vu / (vu + numpy.exp(-lamda)))) - prob_pi
   gradient[n_param_X:n_param] = modelUT.dot(vec_U)

   vec_X = - numpy.where(y_train > 0.0, lamda, lamda * (1.0 - (lamda - 1.0) * vel) / vel2)
   hessian[0:n_param_X, 0:n_param_X] = modelXT.dot((vec_X.reshape((n_sample,1)) * modelX))

   vec_U = numpy.where(y_train > 0.0, 0.0, vel / vel2) - prob_pi * (1.0 - prob_pi)
   hessian[n_param_X:n_param, n_param_X:n_param] = modelUT.dot((vec_U.values.reshape((n_sample,1)) * modelU))

   vec_XU = numpy.where(y_train > 0.0, 0.0, lamda * vel / vel2)
   hessian[0:n_param_X, n_param_X:n_param] = modelXT.dot((vec_XU.reshape((n_sample,1)) * modelU))
   hessian[n_param_X:n_param, 0:n_param_X] = numpy.transpose(hessian[0:n_param_X, n_param_X:n_param])

   orig_diag = numpy.diag(hessian)
   invhessian, aliasParam, nonAliasParam = Regression.SWEEPOperator (n_param, hessian, orig_diag, sweepCol = range(n_param), tol = tolSweep)
   delta = numpy.matmul(-invhessian, gradient)
   step = 1.0
   for iStep in range(maxStep):
      beta_next = beta - step * delta[0:n_param_X]
      nu_X = o_train_X + modelX.dot(beta_next)
      lamda = numpy.exp(nu_X)

      gamma_next = gamma - step * delta[n_param_X: n_param]
      nu_U = o_train_U + modelU.dot(gamma_next)
      vu = numpy.exp(nu_U)

      llk_next = numpy.sum(numpy.where(y_train > 0.0, (y_train * nu_X - lamda - constLLK), (numpy.log(vu + numpy.exp(-lamda)))))
      llk_next = llk_next - numpy.sum(numpy.log(1.0 + vu))

      if ((llk_next - llk) > - tolLLK):
         break
      else:
         step = 0.5 * step

   diff_beta = beta_next - beta
   eps_beta = numpy.linalg.norm(diff_beta)
   diff_gamma = gamma_next - gamma
   eps_gamma = numpy.linalg.norm(diff_gamma)
   llk = llk_next
   beta = beta_next
   gamma = gamma_next
   itList = [it+1, llk, iStep]
   itList.extend(beta)
   itList.extend(gamma)
   iterTable.append(itList)
   if (eps_beta < tolParameter and eps_gamma < tolParameter):
      break

# Final parameter estimates and associated statistics
nu_X = o_train_X + modelX.dot(beta)
lamda = numpy.exp(nu_X)

nu_U = o_train_U + modelU.dot(gamma)
vu = numpy.exp(nu_U)

prob_pi = vu / (1.0 + vu)
y_pred = lamda / (1.0 + vu)

it_name = ['Iteration', 'Log-Likelihood', 'N Steps']
it_name.extend(modelX.columns)
it_name.extend(modelU.columns)
outTable = pandas.DataFrame(iterTable, columns = it_name)

# Final covariance matrix
stderr = numpy.sqrt(numpy.diag(invhessian))
z95 = norm.ppf(0.975)

# Final parameter estimates
stderr_Beta = stderr[0:n_param_X]
stderr_Gamma = stderr[n_param_X:n_param]

outBeta = pandas.DataFrame(beta, index = param_name_X, columns = ['Estimate'])
outBeta.insert(0, 'Class', 'Beta')
outBeta['Standard Error'] = stderr_Beta
outBeta['Lower 95% CI'] = beta - z95 * stderr_Beta
outBeta['Upper 95% CI'] = beta + z95 * stderr_Beta
outBeta['Exponentiated'] = numpy.exp(beta)

outGamma = pandas.DataFrame(gamma, index = param_name_U, columns = ['Estimate'])
outGamma.insert(0, 'Class', 'Gamma')
outGamma['Standard Error'] = stderr_Gamma
outGamma['Lower 95% CI'] = gamma - z95 * stderr_Gamma
outGamma['Upper 95% CI'] = gamma + z95 * stderr_Gamma

outCorb_Beta = pandas.DataFrame((invhessian[0:n_param_X,0:n_param_X] / numpy.outer(stderr_Beta, stderr_Beta)),
                                index = param_name_X, columns = param_name_X)

outCorb_Gamma = pandas.DataFrame((invhessian[n_param_X:n_param,n_param_X:n_param] / numpy.outer(stderr_Gamma, stderr_Gamma)),
                                index = param_name_U, columns = param_name_U)

outCorb_BG = pandas.DataFrame((invhessian[0:n_param_X,n_param_X:n_param] / numpy.outer(stderr_Beta, stderr_Gamma)),
                                index = param_name_X, columns = param_name_U)

# Root Mean Squared Error
y_resid = y_train - y_pred
print('Sum of Residuals = ', numpy.sum(y_resid))

mse = numpy.sum(numpy.power(y_resid, 2)) / n_sample

rmse = numpy.sqrt(mse)
print('Root Mean Squared Error = ', rmse)

# Relative Error
relerr = mse / numpy.var(y_train, ddof = 0)
print('Relative Error = ', relerr)

sqcor = numpy.power(numpy.corrcoef(y_train, y_pred), 2)
print('Squared Correlation = ', sqcor[0,1])

# Plot predicted probability of not filing versus number of claims
plt.figure(dpi = 200)
sg = plt.scatter(y_train, prob_pi, c = e_train, marker = 'o')
plt.xlabel('Observed Number of Claims')
plt.ylabel('Probability of Not Filing Claim')
plt.xticks(range(10))
plt.yticks(numpy.arange(0.0, 1.1, 0.1))
plt.grid(axis = 'both')
cbar = plt.colorbar(sg, label = 'Exposure')
cbar.set_ticks(numpy.arange(0.0, 1.1, 0.1))
plt.show()

# Plot predicted number of claims versus observed number of claims
plt.figure(dpi = 200)
sg = plt.scatter(y_train, y_pred, c = e_train, marker = 'o')
plt.xlabel('Observed Number of Claims')
plt.ylabel('Predicted Number of Claims')
plt.xticks(range(10))
plt.grid(axis = 'both')
cbar = plt.colorbar(sg, label = 'Exposure')
cbar.set_ticks(numpy.arange(0.0, 1.1, 0.1))
plt.show()

# Calculate simple residuals
plt.figure(dpi = 200)
sg = plt.scatter(y_train, y_resid, c = e_train, marker = 'o')
plt.xlabel('Observed CLM_COUNT')
plt.ylabel('Simple Residual')
plt.xticks(range(10))
plt.grid(axis = 'both')
plt.colorbar(sg, label = 'Exposure')
plt.show()

# Calculate Pearson residuals
pearsonResid = numpy.where(y_pred > 0.0, y_resid / numpy.sqrt(y_pred), numpy.NaN)

plt.figure(dpi = 200)
sg = plt.scatter(y_train, pearsonResid, c = e_train, marker = 'o')
plt.xlabel('Observed CLM_COUNT')
plt.ylabel('Pearson Residual')
plt.xticks(range(10))
plt.grid(axis = 'both')
plt.colorbar(sg, label = 'Exposure')
plt.show()

# Calculate deviance residuals
yPos = -numpy.log(1.0 - prob_pi) - y_train * numpy.log(lamda) + lamda + constLLK
yZero = -numpy.log(prob_pi + (1.0 - prob_pi) * numpy.exp(-lamda))
dR2 = numpy.where(y_train > 0.0, yPos, yZero)
devResid = numpy.where(y_train > y_pred, 1.0, -1.0) * numpy.where(dR2 > 0.0, numpy.sqrt(2.0 * dR2), 0.0)

plt.figure(dpi = 200)
sg = plt.scatter(y_train, devResid, c = e_train, marker = 'o')
plt.xlabel('Observed CLM_COUNT')
plt.ylabel('Deviance Residual')
plt.xticks(range(10))
plt.grid(axis = 'both')
plt.colorbar(sg, label = 'Exposure')
plt.show()

# Pearson Chi-Squares
pearson_chisq = numpy.sum(numpy.power(pearsonResid, 2.0))
deviance_chisq = 2.0 * numpy.sum(numpy.power(devResid, 2.0))

df_chisq = n_sample - len(nonAliasParam)

pearson_sig = chi2.sf(pearson_chisq, df_chisq)
deviance_sig = chi2.sf(deviance_chisq, df_chisq)
