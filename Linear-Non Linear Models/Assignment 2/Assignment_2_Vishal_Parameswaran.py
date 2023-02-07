import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas
import sys
import Regression
import seaborn as sns
import math
from scipy.stats import chi2
from sklearn.metrics import mean_squared_error

np.set_printoptions(precision=8, threshold=sys.maxsize)
np.set_printoptions(linewidth=np.inf)
pandas.set_option('display.max_columns', None)
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', None)


def create_categorical_value(dfs, column_name):
    u = dfs[column_name].astype('category')
    u_freq = u.value_counts(ascending=True)
    pm = u.cat.reorder_categories(list(u_freq.index))
    term_df_type = pandas.get_dummies(pm)
    return term_df_type


def run_against_all(dfs, init_cols, categorical_cols, previous_train, output_train, exposure_train, ll, dv):
    step_detail = []
    output_list = []
    for i in init_cols:
        if i in categorical_cols:
            new_train = previous_train.join(create_categorical_value(dfs, i), rsuffix=i)
        else:
            new_train = previous_train.join(dfs[i], rsuffix=i)
        regression_output = Regression.PoissonRegression(new_train, output_train, exposure_train)
        ll_1 = regression_output[3]
        dv_1 = len(regression_output[4])
        dev_chisq = 2 * (ll_1 - ll)
        deviance_df = dv_1 - dv
        dev_sig = chi2.sf(dev_chisq, deviance_df)
        step_detail.append([str(i), dv_1, ll_1, dev_chisq, deviance_df, dev_sig])
        output_list.append(regression_output)
    step_df = pd.DataFrame(step_detail)
    chosen_value = step_df[step_df[5] == step_df[5].min()]
    chosen_output = output_list[step_df.index[step_df[5] == step_df[5].min()].tolist()[0]]
    return chosen_value.values, chosen_output


def forward_selector(data, target_col, exposure_col, init_cols, cat_columns):
    # First Run
    target_train = data[target_col]
    exposure_train = np.log(data[exposure_col])
    intercept_train = data[[target_col]]
    init = init_cols
    intercept_train.insert(0, 'Intercept', 1.0)
    intercept_train = intercept_train.drop(columns=target_col)
    intercept_result = Regression.PoissonRegression(X_train, y_train, o_train)
    step_summary = []
    ll = intercept_result[3]
    dv = len(intercept_result[4])
    dev_sig = 0
    threshold = 0.01
    value_outputs = None
    step_summary.append(['Intercept', dv, ll, np.nan, np.nan, np.nan])
    while dev_sig < threshold:
        run, s = run_against_all(data, init, cat_columns, intercept_train, target_train, exposure_train, ll, dv)
        run = run[0]
        ll = run[2]
        dv = run[1]
        dev_sig = run[5]
        if dev_sig < threshold:
            value_outputs = s
            if run[0] in cat_columns:
                intercept_train = intercept_train.join(create_categorical_value(data, run[0]), rsuffix=str(run[0]))
            else:
                intercept_train = intercept_train.join(data[run[0]], rsuffix=str(run[0]))
            init.remove(run[0])
            step_summary.append([run[0], dv, ll, run[3], run[4], run[5]])
    step_summary = pd.DataFrame(step_summary)
    return step_summary, value_outputs, intercept_result


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
    distance_correlation_coeff = math.sqrt(emp_dist_xy / (math.sqrt(emp_dist_x * emp_dist_y)))
    return distance_correlation_coeff


def find_metrics(y_actual, y_predicted):
    rmse = mean_squared_error(y_true=y_actual, y_pred=y_predicted, squared=False)
    observed_mean = np.mean(y_actual)
    predicted_mean = np.mean(y_predicted)
    rel_error = np.sum((y_actual - y_predicted) ** 2) / np.sum((y_actual - observed_mean) ** 2)
    dist_corr = distance_corr(y_actual, y_predicted)
    # r_sq = r2_score(y_true,y_pred)
    r_sq = np.sum(((y_predicted - predicted_mean) * (y_actual - observed_mean)) ** 2) / (
            np.sum((y_predicted - predicted_mean) ** 2) * np.sum((y_actual - observed_mean) ** 2))
    y_df = pd.DataFrame(y_actual, columns=["Y_True"])
    y_df["Y_Train"] = y_predicted
    pearson_corr = y_df[["Y_True", "Y_Train"]].corr(method="pearson")
    output_metric = pd.DataFrame({"RMSE": [round(rmse, 8)], "Relative Error": [round(rel_error, 8)],
                                  "Pearson Correlation": [round(pearson_corr.iloc[0][1], 8)],
                                  "Distance Correlation": [round(dist_corr, 8)], "R Squared metrics": [round(r_sq, 8)]})
    return output_metric


target = 'CLM_COUNT'
exposure = 'EXPOSURE'
int_pred = ["CAR_TYPE", "CAR_USE", "EDUCATION", "GENDER", "MSTATUS", "PARENT1", "RED_CAR", "REVOKED", "URBANICITY",
            'HOMEKIDS', 'KIDSDRIV', "AGE", "BLUEBOOK", "CAR_AGE", "HOME_VAL", "INCOME", "YOJ", "MVR_PTS", "TIF",
            "TRAVTIME"]
cat_cols = ["CAR_TYPE", "CAR_USE", "EDUCATION", "GENDER", "MSTATUS", "PARENT1", "RED_CAR", "REVOKED", "URBANICITY"]
claim_history = pandas.read_excel('claim_history.xlsx')
claim_history[["BLUEBOOK", "HOME_VAL", "INCOME"]] = claim_history[["BLUEBOOK", "HOME_VAL", "INCOME"]] / 1000

train_data = claim_history[claim_history['EXPOSURE'] > 0.0]
train_data = train_data[[target] + [exposure] + int_pred]  # Only necessary
train_data = train_data.dropna().reset_index()  # Remove missing values

# Question 1.1
q1a = sns.histplot(train_data["CLM_COUNT"])
q1a.set_ylabel("Frequency")
q1a.set_xlabel("Claim Count")

# For Question 1.1 If we have to print the frequencies based on the original claim data before the intercept model
q2a = sns.histplot(claim_history["CLM_COUNT"])
q2a.set_ylabel("Frequency")
q2a.set_xlabel("Claim Count")

# Training on the intercept Alone
n_sample = train_data.shape[0]
y_train = train_data[target]
o_train = np.log(train_data[exposure])
X_train = train_data[[target]]
X_train.insert(0, 'Intercept', 1.0)
X_train = X_train.drop(columns=target)
result = Regression.PoissonRegression(X_train, y_train, o_train)
llk = result[3]
nonAliasParam = result[4]
outIterationTable = result[5]
y_pred = result[6]
aic = -2 * llk + 2 * len(nonAliasParam)
bic = -2 * llk + len(nonAliasParam) * math.log(n_sample)

print("###############################################################################################################")
# Question 1.2
print("Log Liklehood = " + str(llk), "AIC = " + str(aic), "BIC = " + str(bic))

# Training on all the other Predictors
int_pred = ["CAR_TYPE", "CAR_USE", "EDUCATION", "GENDER", "MSTATUS", "PARENT1", "RED_CAR", "REVOKED", "URBANICITY",
            'HOMEKIDS', 'KIDSDRIV', "AGE", "BLUEBOOK", "CAR_AGE", "HOME_VAL", "INCOME", "YOJ", "MVR_PTS", "TIF",
            "TRAVTIME"]
df, outps, train = forward_selector(train_data, target, exposure, int_pred, cat_cols)

print("###############################################################################################################")
# Question 2.1
print(df)

print("###############################################################################################################")
# Question 2.2
print(outps[0])

aic = -2 * outps[3] + 2 * len(outps[4])
bic = -2 * outps[3] + len(outps[4]) * math.log(n_sample)

print("###############################################################################################################")
# Question 2.3
print("AIC = " + str(aic), "BIC = " + str(bic))

print("###############################################################################################################")
# Question 3.1
print(find_metrics(train_data[target].values, train[6]))

print("###############################################################################################################")
# Question 3.2
print(find_metrics(train_data[target].values, outps[6]))

pred_y = outps[6]
true_y = train_data[target].values
pearson_chi_statistic = np.sum((true_y - pred_y) ** 2 / pred_y)
dof = n_sample - len(outps[4])
di = -((true_y * np.log(pred_y / true_y)) + (true_y - pred_y))
dR2 = np.where(true_y > 0.0, di, 0)
devResid = np.where(true_y > pred_y, 1.0, -1.0) * np.where(dR2 > 0.0, np.sqrt(2.0 * dR2), 0.0)
deviance_chisq = np.sum(np.power(devResid, 2.0))
pearson_sig = chi2.sf(pearson_chi_statistic, dof)
deviance_sig = chi2.sf(deviance_chisq, dof)
comparative_output = pd.DataFrame(
    {"Type": ["Pearson", "Deviance"], "Statistic": [pearson_chi_statistic, deviance_chisq],
     "Degrees of Freedom": [dof, dof], "Significance": [pearson_sig, deviance_sig]})
print("###############################################################################################################")
# Question 3.3
print(comparative_output)

# Question 4.1
y_resid = true_y - pred_y
pearsonResid = np.where(pred_y > 0.0, y_resid / np.sqrt(pred_y), np.NaN)
plt.figure(dpi=200)
sg = plt.scatter(true_y, pearsonResid, c=train_data[exposure], marker='o', cmap="viridis")
plt.xlabel('Observed CLM_COUNT')
plt.ylabel('Pearson Residual')
plt.xticks(range(10))
plt.grid(axis='both')
plt.colorbar(sg, label='Exposure')
plt.grid()
plt.show()

# Question 4.2
plt.figure(dpi=200)
sg = plt.scatter(true_y, devResid, c=train_data[exposure], marker='o', cmap="viridis")
plt.xlabel('Observed CLM_COUNT')
plt.ylabel('Deviance Residual')
plt.xticks(range(10))
plt.grid(axis='both')
plt.colorbar(sg, label='Exposure')
plt.grid()
plt.show()
