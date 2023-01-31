# Name: Regression.py
# Creation Date: May 24, 2020
# Last Update: May 10, 2022
# Author: Ming-Long Lam

import numpy
import pandas

from scipy.special import digamma, gammaln
from scipy.stats import norm

def SWEEPOperator (pDim, inputM, origDiag, sweepCol = None, tol = 1e-7):
    ''' Implement the SWEEP operator

    Parameter
    ---------
    pDim: dimension of matrix inputM, integer greater than one
    inputM: a square and symmetric matrix, numpy array
    origDiag: the original diagonal elements before any SWEEPing
    sweepCol: a list of columns numbers to SWEEP
    tol: singularity tolerance, positive real

    Return
    ------
    A: negative of a generalized inverse of input matrix
    aliasParam: a list of aliased rows/columns in input matrix
    nonAliasParam: a list of non-aliased rows/columns in input matrix
    '''

    if (sweepCol is None):
        sweepCol = range(pDim)

    aliasParam = []
    nonAliasParam = []

    A = numpy.copy(inputM)
    ANext = numpy.zeros((pDim,pDim))

    for k in sweepCol:
        Akk = A[k,k]
        pivot = tol * abs(origDiag[k])
        if (not numpy.isinf(Akk) and abs(Akk) >= pivot and pivot > 0.0):
            nonAliasParam.append(k)
            ANext = A - numpy.outer(A[:, k], A[k, :]) / Akk
            ANext[:, k] = A[:, k] / abs(Akk)
            ANext[k, :] = ANext[:, k]
            ANext[k, k] = -1.0 / Akk
        else:
            aliasParam.append(k)
            ANext[:,k] = numpy.zeros(pDim)
            ANext[k, :] = numpy.zeros(pDim)
        A = ANext
    return (A, aliasParam, nonAliasParam)

def create_interaction (df1, df2):
    ''' Return the columnwise product of two dataframes (must have same number of rows)

    Parameter
    ---------
    df1: first input data frame
    df2: second input data frame

    Return
    ------
    outDF: the columnwise product of two dataframes
    '''

    name1 = df1.columns
    name2 = df2.columns
    outDF = pandas.DataFrame()
    for col1 in name1:
        outName = col1 + ' * ' + name2
        outDF[outName] = df2.multiply(df1[col1], axis = 'index')
    return(outDF)

def paste_interaction (interactName):
    ipos = interactName.find('*')
    name1 = interactName[:ipos].strip()
    name2 = interactName[(ipos+1):].strip()
    return (name1, name2)

def binary_model_metric (target, valueEvent, valueNonEvent, predProbEvent, eventProbThreshold = 0.5):
    '''Calculate metrics for a binary classification model

    Parameter
    ---------
    target: Panda Series that contains values of target variable
    valueEvent: Formatted value of target variable that indicates an event
    valueNonEvent: Formatted value of target variable that indicates a non-event
    predProbEvent: Panda Series that contains predicted probability that the event will occur
    eventProbThreshold: Threshold for event probability to indicate a success

    Return
    ------
    outSeries: Pandas Series that contain the following statistics
               ASE: Average Squared Error
               RASE: Root Average Squared Error
               MCE: Misclassification Rate
               AUC: Area Under Curve
    '''

    # Number of observations
    nObs = len(target)

    # Aggregate observations by the target values and the predicted probabilities
    aggrProb = pandas.crosstab(predProbEvent, target, dropna = True)

    # Calculate the root average square error
    ase = (numpy.sum(aggrProb[valueEvent] * (1.0 - aggrProb.index)**2) +
           numpy.sum(aggrProb[valueNonEvent] * (0.0 - aggrProb.index)**2)) / nObs
    if (ase > 0.0):
        rase = numpy.sqrt(ase)
    else:
        rase = 0.0

    # Calculate the misclassification error rate
    nFP = numpy.sum(aggrProb[valueNonEvent].iloc[aggrProb.index >= eventProbThreshold])
    nFN = numpy.sum(aggrProb[valueEvent].iloc[aggrProb.index < eventProbThreshold])
    mce = (nFP + nFN) / nObs

    # Calculate the number of concordant, discordant, and tied pairs
    nConcordant = 0.0
    nDiscordant = 0.0
    nTied = 0.0

    # Loop over the predicted event probabilities from the Event column
    predEP = aggrProb.index
    eventFreq = aggrProb[valueEvent]

    for i in range(len(predEP)):
        eProb = predEP[i]
        eFreq = eventFreq.loc[eProb]
        if (eFreq > 0.0):
            nConcordant = nConcordant + numpy.sum(eFreq * aggrProb[valueNonEvent].iloc[eProb > aggrProb.index])
            nDiscordant = nDiscordant + numpy.sum(eFreq * aggrProb[valueNonEvent].iloc[eProb < aggrProb.index])
            nTied = nTied + numpy.sum(eFreq * aggrProb[valueNonEvent].iloc[eProb == aggrProb.index])

    auc = 0.5 + 0.5 * (nConcordant - nDiscordant) / (nConcordant + nDiscordant + nTied)

    outSeries = pandas.Series({'ASE': ase, 'RASE': rase, 'MCE': mce, 'AUC': auc})
    return(outSeries)

def curve_coordinates (target, valueEvent, valueNonEvent, predProbEvent):
    '''Calculate coordinates of the Receiver Operating Characteristics (ROC) curve and
    the Precision Recall (PR) curve

    Classification Convention
    -------------------------
    An observation is classified as Event if the predicted event probability is
    greater than or equal to a given threshold value.

    Parameter
    ---------
    target: Panda Series that contains values of target variable
    valueEvent: Formatted value of target variable that indicates an event
    valueNonEvent: Formatted value of target variable that indicates a non-event
    predProbEvent: Panda Series that contains predicted probability that the event will occur

    Return
    ------
    outCurve: Pandas dataframe for the curve coordinates
              Threshold: Event probability threshold of the coordinates
              Sensitivity: Sensitivity coordinate
              OneMinusSpecificity: 1 - Specificity coordinate
              Precision: Precision coordinate
              Recall: Recall coordinate
    '''

    # Aggregate observations by the target values and the predicted probabilities
    aggrProb = pandas.crosstab(predProbEvent, target, dropna = True)
    threshValue = aggrProb.index
    n_thresh = len(threshValue)
    curve_coord = numpy.empty((n_thresh,6), dtype=float)

    # Find out the number of events and non-events
    n_event = numpy.sum(aggrProb[valueEvent])
    n_nonevent = numpy.sum(aggrProb[valueNonEvent])

    for i in range(n_thresh):
        thresh = threshValue[i]
        nTP = numpy.sum(aggrProb[valueEvent].iloc[threshValue >= thresh])
        nFP = numpy.sum(aggrProb[valueNonEvent].iloc[threshValue >= thresh])

        Sensitivity = nTP / n_event
        OneMinusSpecificity = nFP / n_nonevent
        Precision = nTP / (nTP + nFP)
        Recall = Sensitivity
        F1Score = 2.0 / (1.0 / Precision + 1.0 / Recall)

        curve_coord[i,:] = [thresh, Sensitivity, OneMinusSpecificity, Precision, Recall, F1Score]

    outCurve = pandas.DataFrame(curve_coord, index = range(n_thresh),
                                columns = ['Threshold', 'Sensitivity', 'OneMinusSpecificity', 'Precision', 'Recall', 'F1 Score'])

    return (outCurve)

def LinearRegression (X, y):
    ''' Train a linear regression model

    Parameter
    ---------
    X: A Pandas DataFrame, rows are observations, columns are regressors
    y: A Pandas Series, rows are observations of the response variable

    Return
    ------
    b: an array of regression coefficient
    residual_SS: residual sum of squares
    XtX_Ginv: a generalized inverse of the XtX matrix
    aliasParam: a list of aliased rows/columns in input matrix
    nonAliasParam: a list of non-aliased rows/columns in input matrix
    '''

    # X: A Pandas DataFrame, rows are observations, columns are regressors
    # y: A Pandas Series, rows are observations of the response variable

    Z = X.join(y)
    n_sample = Z.shape[0]
    n_param = Z.shape[1] - 1

    ZtZ = Z.transpose().dot(Z)
    diag_ZtZ = numpy.diagonal(ZtZ)
    eps_double = numpy.finfo(numpy.float64).eps
    tol = numpy.sqrt(eps_double)

    ZtZ_transf, aliasParam, nonAliasParam = SWEEPOperator ((n_param+1), ZtZ, diag_ZtZ, sweepCol = range(n_param), tol = tol)

    b = ZtZ_transf[0:n_param, n_param]
    b[aliasParam] = 0.0

    XtX_Ginv = - ZtZ_transf[0:n_param, 0:n_param]
    XtX_Ginv[:, aliasParam] = 0.0
    XtX_Ginv[aliasParam, :] = 0.0

    residual_SS = ZtZ_transf[n_param, n_param]

    return (b, residual_SS, XtX_Ginv, aliasParam, nonAliasParam)

def PoissonRegression (X, y, offset = None, maxIter = 20, maxStep = 5, tolLLK = 1e-3, tolBeta = 1e-10, tolSweep = 1e-7):
    ''' Train a Generalized Linear Model with Poisson distribution and Logarithm link function

    Parameter
    ---------
    X: A Pandas DataFrame, rows are observations, columns are regressors
    y: A Pandas Series, rows are observations of the response variable
    offset: A Pandas Series of offset values
    maxIter: Maximum number of iterations
    maxStep: Maximum number of step-halving
    tolLLK: Minimum absolute difference to get a successful step-halving
    tolBeta: Maximum absolute difference between successive sets of parameter estimates to call convergence
    tolSweep: Tolerance for SWEEP Operator

    Return
    ------
    outCoefficient: a 2D array of regression coefficients, standard errors, and confidence interval
    outCovb: a 2D array of covariance matrix of regression coefficients
    outCorb: a 2D array of correlation matrix of regression coefficients
    llk: log-likelihood value
    nonAliasParam: a list of non-aliased rows/columns in input matrix
    outIterationTable: a 2D array of iteration history table
    y_pred: a 1D array of predicted target values
    '''

    modelX = X.copy()
    n_sample = modelX.shape[0]
    n_param = modelX.shape[1]
    param_name = modelX.columns

    modelXT = modelX.transpose()

    # Precompute the ln(y!)
    constLLK = gammaln(y+1.0)

    # Initialize arrays
    beta = numpy.zeros(n_param)
    y_mean = numpy.mean(y)
    beta[0] = numpy.log(y_mean)
    if (offset is not None):
       beta[0] = beta[0] - numpy.mean(offset)

    nu = modelX.dot(beta)
    if (offset is not None):
        nu = offset + nu
    y_pred = numpy.exp(nu)
    llk = numpy.sum(y * nu - y_pred - constLLK)

    # Prepare the iteration history table (Iteration #, Log-Likelihood, N Step-Halving, Beta)
    itList = [0, llk, 0]
    itList.extend(beta)
    iterTable = [itList]

    for it in range(maxIter):
        gradient = modelXT.dot((y - y_pred))
        hessian = - modelXT.dot((y_pred.values.reshape((n_sample,1)) * modelX))
        orig_diag = numpy.diag(hessian)
        invhessian, aliasParam, nonAliasParam = SWEEPOperator (n_param, hessian, orig_diag, sweepCol = range(n_param), tol = tolSweep)
        delta = numpy.matmul(-invhessian, gradient)
        step = 1.0
        for iStep in range(maxStep):
            beta_next = beta - step * delta
            nu_next = numpy.dot(modelX, beta_next)
            if (offset is not None):
                nu_next = offset + nu_next
            y_pred_next = numpy.exp(nu_next)
            llk_next = numpy.sum(y * nu_next - y_pred_next - constLLK)
            if ((llk_next - llk) > - tolLLK):
                break
            else:
                step = 0.5 * step
        diffBeta = beta_next - beta
        llk = llk_next
        beta = beta_next
        y_pred = y_pred_next
        itList = [it+1, llk, iStep]
        itList.extend(beta)
        iterTable.append(itList)
        if (numpy.linalg.norm(diffBeta) < tolBeta):
            break

    it_name = ['Iteration', 'Log-Likelihood', 'N Step-Halving']
    it_name.extend(param_name)
    outIterationTable = pandas.DataFrame(iterTable, columns = it_name)

    # Final covariance matrix
    stderr = numpy.sqrt(numpy.diag(invhessian))
    z95 = norm.ppf(0.975)

    # Final parameter estimates
    outCoefficient = pandas.DataFrame(beta, index = param_name, columns = ['Estimate'])
    outCoefficient['Standard Error'] = stderr
    outCoefficient['Lower 95% CI'] = beta - z95 * stderr
    outCoefficient['Upper 95% CI'] = beta + z95 * stderr
    outCoefficient['Exponentiated'] = numpy.exp(beta)

    outCovb = pandas.DataFrame(invhessian, index = param_name, columns = param_name)

    temp_m1_ = numpy.outer(stderr, stderr)
    outCorb = pandas.DataFrame(numpy.divide(invhessian, temp_m1_, out = numpy.zeros_like(invhessian), where = (temp_m1_ != 0.0)),
                               index = param_name, columns = param_name)

    return ([outCoefficient, outCovb, outCorb, llk, nonAliasParam, outIterationTable, y_pred])

def BinaryLogisticRegression (X, y, offset = None, maxIter = 20, maxStep = 5, tolLLK = 1e-3, tolBeta = 1e-10, tolSweep = 1e-7):
    ''' Train a Generalized Linear Model with Bernoulli distribution and Logit link function

    Parameter
    ---------
    X: A Pandas DataFrame, rows are observations, columns are regressors
    y: A Pandas Series, rows are observations of the response variable (1 for Event, 0 otherwise)
    offset: A Pandas Series of offset values
    maxIter: Maximum number of iterations
    maxStep: Maximum number of step-halving
    tolLLK: Minimum absolute difference to get a successful step-halving
    tolBeta: Maximum absolute difference between successive sets of parameter estimates to call convergence
    tolSweep: Tolerance for SWEEP Operator

    Return
    ------
    outCoefficient: a 2D array of regression coefficients, standard errors, and confidence interval
    outCovb: a 2D array of covariance matrix of regression coefficients
    outCorb: a 2D array of correlation matrix of regression coefficients
    llk: log-likelihood value
    nonAliasParam: a list of non-aliased rows/columns in input matrix
    outIterationTable: a 2D array of iteration history table
    y_predprob: a 2D array of predicted probabilities
    '''

    modelX = X.copy()
    n_sample = modelX.shape[0]
    n_param = modelX.shape[1]
    param_name = modelX.columns

    modelXT = modelX.transpose()

    # Initialize predicted probabilities
    pEvent = numpy.mean(y)
    pNonEvent = 1.0 - pEvent
    odds = pEvent / pNonEvent
    y_p1 = numpy.full(n_sample, pEvent)
    beta = numpy.zeros(n_param)
    beta[0] = numpy.log(odds)
    nu = modelX.dot(beta)
    if (offset is not None):
        nu = offset + nu
    llk = numpy.sum(y * nu + numpy.log(pNonEvent))

    # Prepare the iteration history table (Iteration #, Log-Likelihood, N Step-Halving, Beta)
    itList = [0, llk, 0, beta]
    iterTable = [itList]

    for it in range(maxIter):
        gradient = numpy.dot(modelXT, (y - y_p1))
        dispersion = pandas.DataFrame(y_p1 * (1.0 - y_p1))
        hessian = - numpy.dot(modelXT, (dispersion.values.reshape((n_sample,1)) * modelX))
        orig_diag = numpy.diag(hessian)
        invhessian, aliasParam, nonAliasParam = SWEEPOperator (n_param, hessian, orig_diag, sweepCol = range(n_param), tol = tolSweep)
        delta = numpy.matmul(-invhessian, gradient)
        step = 1.0
        for iStep in range(maxStep):
            beta_next = beta - step * delta
            nu_next = numpy.dot(modelX, beta_next)
            if (offset is not None):
                nu_next = offset + nu_next
            odds = numpy.exp(nu_next)
            y_p0 = 1.0 / (1.0 + odds)
            llk_next = numpy.sum(y * nu_next + numpy.log(y_p0))
            if ((llk_next - llk) > - tolLLK):
                break
            else:
                step = 0.5 * step

        diffBeta = beta_next - beta
        llk = llk_next
        beta = beta_next
        y_p1 = 1.0 - y_p0
        itList = [it+1, llk, iStep, beta]
        iterTable.append(itList)
        if (numpy.linalg.norm(diffBeta) < tolBeta):
            break

    it_name = ['Iteration', 'Log-Likelihood', 'N Step-Halving', 'Parameters']
    outIterationTable = pandas.DataFrame(iterTable, columns = it_name)

    # Final covariance matrix
    stderr = numpy.sqrt(numpy.diag(invhessian))
    z95 = norm.ppf(0.975)

    # Final parameter estimates
    outCoefficient = pandas.DataFrame(beta, index = param_name, columns = ['Estimate'])
    outCoefficient['Standard Error'] = stderr
    outCoefficient['Lower 95% CI'] = beta - z95 * stderr
    outCoefficient['Upper 95% CI'] = beta + z95 * stderr

    outCovb = pandas.DataFrame(invhessian, index = param_name, columns = param_name)

    temp_m1_ = numpy.outer(stderr, stderr)
    outCorb = pandas.DataFrame(numpy.divide(invhessian, temp_m1_, out = numpy.zeros_like(invhessian), where = (temp_m1_ != 0.0)),
                               index = param_name, columns = param_name)

    y_cats = y.astype('category').cat.categories
    y_predprob = pandas.DataFrame(columns = y_cats)
    y_predprob[y_cats[0]] = y_p0
    y_predprob[y_cats[1]] = y_p1

    return ([outCoefficient, outCovb, outCorb, llk, nonAliasParam, outIterationTable, y_predprob])

def solve4Alpha (c, maxIter = 100, epsilon = 1e-10):
    ''' Use bisection search to solve this equation for alpha:
        log(alpha) - digamma(alpha) = c

    Parameter
    ---------
    c: A positive value

    Return
    ------
    alpha: Solution of the equation, a positive value
    '''

    # Find a0 such that f0 is greater than or equal to c
    a0 = 0.5
    while True:
        f0 = numpy.log(a0) - digamma(a0)
        if (f0 < c):
            a0 = a0 / 2.0
        else:
            break

    # Find a1 such that f1 is less than or equal to c
    a1 = 2.0
    while True:
        f1 = numpy.log(a1) - digamma(a1)
        if (f1 > c):
            a1 = a1 * 2.0
        else:
            break

    # Update the end-points
    for iIter in range(maxIter):
        alpha = (a0 + a1) / 2.0
        func = numpy.log(alpha) - digamma(alpha)
        if (abs(func-c) > epsilon):
            if (func > c):
                a0 = alpha
            else:
                a1 = alpha
        else:
            break

    return (alpha)

def GammaRegression (X, y, offset = None, maxIter = 20, maxStep = 5, tolLLK = 1e-3, tolBeta = 1e-10, tolSweep = 1e-7):
    ''' Train a Generalized Linear Model with Gamma distribution and Logarithm link function

    Parameter
    ---------
    X: A Pandas DataFrame, rows are observations, columns are regressors
    y: A Pandas Series, rows are observations of the response variable
    offset: A Pandas Series of offset values
    maxIter: Maximum number of iterations
    maxStep: Maximum number of step-halving
    tolLLK: Minimum absolute difference to get a successful step-halving
    tolBeta: Maximum absolute difference between successive sets of parameter estimates to call convergence
    tolSweep: Tolerance for SWEEP Operator

    Return
    ------
    outCoefficient: a 2D array of regression coefficients, standard errors, and confidence interval
    outCovb: a 2D array of covariance matrix of regression coefficients
    outCorb: a 2D array of correlation matrix of regression coefficients
    llk: log-likelihood value
    nonAliasParam: a list of non-aliased rows/columns in input matrix
    outIterationTable: a 2D array of iteration history table
    y_pred: a 1D array of predicted target values
    alpha: the shape parameter
    '''

    modelX = X.copy()
    n_sample = modelX.shape[0]
    n_param = modelX.shape[1]
    param_name = modelX.columns

    modelXT = modelX.transpose()

    # Precompute the ln(y)
    y_log = numpy.log(y)

    # Initialize beta array and scale parameter alpha
    beta = numpy.zeros(n_param)
    beta[0] = numpy.log(numpy.mean(y))
    nu = modelX.dot(beta)
    if (offset is not None):
        nu = offset + nu
    y_pred = numpy.exp(nu)
    rvec = numpy.divide(y, y_pred)
    c = numpy.mean(rvec - numpy.log(rvec)) - 1.0
    alpha = solve4Alpha(c)
    uvec = - alpha * (rvec + numpy.log(y_pred)) + (alpha - 1.0) * y_log
    llk = numpy.sum(uvec) + n_sample * (alpha * numpy.log(alpha) - gammaln(alpha))

    # Prepare the iteration history table (Iteration #, Log-Likelihood, N Iteration, N Step-Halving, Convergence, alpha, Beta)
    itList = [0, llk, 0, 0, numpy.nan, alpha, beta]
    iterTable = [itList]

    for it in range(maxIter):
        rvec = numpy.divide(y, y_pred)
        gradient = modelXT.dot((rvec - 1.0))
        hessian = - modelXT.dot((rvec.values.reshape((n_sample,1)) * modelX))
        orig_diag = numpy.diag(hessian)
        invhessian, aliasParam, nonAliasParam = SWEEPOperator (n_param, hessian, orig_diag, sweepCol = range(n_param), tol = tolSweep)
        delta = numpy.matmul(-invhessian, gradient)
        step = 1.0
        for iStep in range(maxStep):
            beta_next = beta - step * delta
            nu_next = modelX.dot(beta_next)
            y_pred_next = numpy.exp(nu_next)
            rvec = numpy.divide(y, y_pred_next)
            c = numpy.mean(rvec - numpy.log(rvec)) - 1.0
            alpha_next = solve4Alpha(c)
            uvec = - alpha_next * (rvec + numpy.log(y_pred_next)) + (alpha_next - 1.0) * y_log
            llk_next = numpy.sum(uvec) + n_sample * (alpha_next * numpy.log(alpha_next) - gammaln(alpha_next))
            if ((llk_next - llk) > - tolLLK):
                break
            else:
                step = 0.5 * step
        diffBeta = beta_next - beta
        diffBetaNorm = numpy.linalg.norm(diffBeta)
        llk = llk_next
        alpha = alpha_next
        beta = beta_next
        y_pred = y_pred_next
        itList = [it+1, llk, it, iStep, diffBetaNorm, alpha, beta]
        iterTable.append(itList)
        if (diffBetaNorm < tolBeta):
            break

    it_name = ['Iteration', 'Log-Likelihood', 'N Iteration', 'N Step-Halving', 'Criterion', 'Alpha', 'Parameters']
    outIterationTable = pandas.DataFrame(iterTable, columns = it_name)

    # Final covariance matrix
    stderr = numpy.sqrt(numpy.diag(invhessian))
    z95 = norm.ppf(0.975)

    # Final parameter estimates
    outCoefficient = pandas.DataFrame(beta, index = param_name, columns = ['Estimate'])
    outCoefficient['Standard Error'] = stderr
    outCoefficient['Lower 95% CI'] = beta - z95 * stderr
    outCoefficient['Upper 95% CI'] = beta + z95 * stderr
    outCoefficient['Exponentiated'] = numpy.exp(beta)

    outCovb = pandas.DataFrame(invhessian, index = param_name, columns = param_name)

    temp_m1_ = numpy.outer(stderr, stderr)
    outCorb = pandas.DataFrame(numpy.divide(invhessian, temp_m1_, out = numpy.zeros_like(invhessian), where = (temp_m1_ != 0.0)),
                               index = param_name, columns = param_name)

    return ([outCoefficient, outCovb, outCorb, llk, nonAliasParam, outIterationTable, y_pred, alpha])
