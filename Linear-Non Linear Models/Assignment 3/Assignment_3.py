# Imports
import pandas as pd
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
import Regression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#--------------------------------------------------------------------------------------------#

target = 'Churn' #Setting the target variables
int_pred = ["MonthlyCharges","Tenure","TotalCharges"] # Setting the interval predictors
cat_cols = ["Dependents","Gender","MultipleLines","InternetService","PaperlessBilling","Partner","PhoneService","SeniorCitizen", "Contract"] # Setting the categorical predictors
claim_history = pandas.read_excel('Telco-Customer-Churn.xlsx') # Reading the input file
print("\n######################################################################################\n")
print(claim_history.head())

#--------------------------------------------------------------------------------------------#
# Creating the Training Data

train_data = claim_history.copy() 
train_data = train_data.dropna()
train_data[target] = np.where(train_data[target] == "Yes",1,0)

#--------------------------------------------------------------------------------------------#
# Calculating the Overall Odds

yFreq = train_data.groupby(target).size()
overall_odds = yFreq[1] / yFreq[0]
overall_odds

#--------------------------------------------------------------------------------------------#
# Q1.1

print("\n######################################################################################\n")
fig,axs = plt.subplots(math.ceil(len(cat_cols)/3),3, figsize = (18,18))
counter = 0
row_counter = 0
for i in cat_cols:
    xtab = pandas.crosstab(index = train_data[i], columns = train_data[target])
    xtab.reset_index(inplace = True)
    xtab['N'] = xtab[0] + xtab[1]
    xtab['Odds'] = xtab[1] / xtab[0]
    xtab.sort_values(by = 'Odds', inplace = True,ascending=False)
    plot = sns.barplot(xtab,x=i,y="Odds", color = 'firebrick',ax=axs[row_counter][counter])
    plot.axhline(y = overall_odds, color = 'blue', linestyle = '--', label = 'Overall Odds')
    plot.set_xlabel(i)
    plot.set_ylabel('Odds of Churn=1 vs Churn=0')
    plot.grid(axis ='y')
    if counter == 2:
        counter = 0
        row_counter+=1
    else:
        counter+=1
plt.show()

#--------------------------------------------------------------------------------------------#
# Q1.2

print("\n######################################################################################\n")
counter=0
fig,axs = plt.subplots(3,1,figsize = (6,18))
for i in int_pred:
    predMedian = np.median(train_data[i])
    img = sns.boxplot(data=train_data,x = i, y = target, orient="h",ax=axs[counter])
    img.axvline(x = predMedian, color = 'red', linestyle = '--')
    img.set_xlabel(i)
    img.set_ylabel("Churn")
    img.yaxis.grid(True)
    counter+=1
plt.show()
#--------------------------------------------------------------------------------------------#
# Making a Function to calculate the Categorical values for a given column.

def create_categorical_value(dfs, column_name):
    u = dfs[column_name].astype('category')
    u_freq = u.value_counts(ascending=True)
    pm = u.cat.reorder_categories(list(u_freq.index))
    term_df_type = pandas.get_dummies(pm)
    term_df_type = term_df_type.add_suffix("_"+column_name)
    return term_df_type

#--------------------------------------------------------------------------------------------#
# Making a Function to calculate the maximum significant deviance when comparing a model to a list of columns.

def run_against_all(dfs, init_cols, categorical_cols, previous_train, output_train, ll, dv):
    step_detail = []
    output_list = []
    for i in init_cols:
        if i in categorical_cols:
            new_train = previous_train.drop(create_categorical_value(dfs, i).columns,axis=1)
        else:
            new_train = previous_train.drop(i,axis=1)
        regression_output = Regression.BinaryLogisticRegression(new_train, output_train)
        ll_1 = regression_output[3]
        dv_1 = len(regression_output[4])
        dev_chisq = 2 * (ll - ll_1)
        deviance_df = dv - dv_1
        dev_sig = chi2.sf(dev_chisq, deviance_df)
        step_detail.append([str(i), dv_1, ll_1, dev_chisq, deviance_df, dev_sig])
        output_list.append(regression_output)
    step_df = pd.DataFrame(step_detail)
    chosen_value = step_df[step_df[5] == step_df[5].max()]
    if any(step_df[5].isna()):
        chosen_value = step_df[step_df[5].isna()]
    chosen_output = output_list[step_df.index[step_df[5] == step_df[5].max()].tolist()[0]]
    return chosen_value.values, chosen_output

#--------------------------------------------------------------------------------------------#
# Making a Function to perform Backward selection when given a Dataframe, Categorical Columns and Interval Predictors.

def backward_selector(data, target_col, init_cols, cat_columns):
    # First Run
    target_train = data[target_col]
    intercept_train = data[[target_col]]
    init = init_cols
    intercept_train.insert(0, 'Intercept', 1.0)
    intercept_train = intercept_train.drop(columns=target_col)
    intercept_train = intercept_train.join(train_data[init_cols])
    for i in cat_columns:
        intercept_train = intercept_train.join(create_categorical_value(data,i))
    init = init_cols + cat_columns
    intercept_result = Regression.BinaryLogisticRegression(intercept_train, target_train)
    step_summary = []
    ll = intercept_result[3]
    dv = len(intercept_result[4])
    dev_sig = 1
    threshold = 0.01
    value_outputs = None
    step_summary.append(['Intercept', dv, ll, np.nan, np.nan, np.nan])
    
    while dev_sig > threshold or math.isnan(dev_sig):
        run, s = run_against_all(data, init, cat_columns, intercept_train, target_train, ll, dv)
        run = run[0]
        ll = run[2]
        dv = run[1]
        dev_sig = run[5]
        if dev_sig > threshold or math.isnan(dev_sig):
            value_outputs = s
            if run[0] in cat_columns:
                intercept_train = intercept_train.drop(create_categorical_value(data, run[0]).columns,axis=1)
            else:
                intercept_train = intercept_train.drop(run[0],axis=1)
            init.remove(run[0])
            step_summary.append([run[0], dv, ll, run[3], run[4], run[5]])
    step_summary = pd.DataFrame(step_summary)

    return step_summary,value_outputs, intercept_result

#--------------------------------------------------------------------------------------------#
# Running the backward Selection function on the given data.

df,out,init_res = backward_selector(train_data, target, int_pred, cat_cols)

#--------------------------------------------------------------------------------------------#
# Q2.1

print("\n######################################################################################\n")
print(df)

#--------------------------------------------------------------------------------------------#
# Q2.2.

print("\n######################################################################################\n")
parameter_table = out[0]
print(parameter_table)

#--------------------------------------------------------------------------------------------#
# Q2.3.

profile = [1,29,1400,1,0,0,0,0,1,0,1,1,0,0,0,1]
estimates = parameter_table["Estimate"].values
pred_logit = sum([x*y for x,y in zip(estimates,profile)])
pred_odds = np.exp(pred_logit)/(1+np.exp(pred_logit))
print("\n######################################################################################\n")
print(pred_odds)

#--------------------------------------------------------------------------------------------#
# Training an intercept only model to compute the various RSquared values

init_intercept = train_data[[target]]
init_intercept.insert(0,"Intercept",1)
init_intercept = init_intercept.drop(columns=target)
target_train = train_data[target]
intercept_result = Regression.BinaryLogisticRegression (init_intercept, target_train)
llk_initial = intercept_result[3]
llk_latest = out[3]

y = train_data[target].values
n = len(y)
rsq_mcf = 1 - (llk_latest/llk_initial)
rsq_cox = (2.0 / n) * (llk_initial - llk_latest)
rsq_cox = 1.0 - np.exp(rsq_cox)
upbound = (2.0 / n) * llk_initial
upbound = 1.0 - np.exp(upbound)
rsq_nag = rsq_cox / upbound
predprob_event = out[6][1]
# print(y)
s1 = np.mean(predprob_event[y == 1])
s0 = np.mean(predprob_event[y == 0])
rsq_tju = s1 - s0

#--------------------------------------------------------------------------------------------#
# Q3.1.

outputs = pd.DataFrame({"McFadden" : [rsq_mcf],"Cox-Snell" : [rsq_cox],"Nagelkerke" : [rsq_nag],"Tjur" : [rsq_tju]})
print("\n######################################################################################\n")
print(outputs)

#--------------------------------------------------------------------------------------------#
# Q3.2.

value = Regression.binary_model_metric(target = y,valueEvent = 1,valueNonEvent = 0, predProbEvent = predprob_event)
auc = value [3]
print("\n######################################################################################\n")
print(" Area Under Curve = " + str(auc))

#--------------------------------------------------------------------------------------------#
# Q3.3.

rase = value[1]
print("\n######################################################################################\n")
print("Root Average Squared Error = "+str(rase))

#--------------------------------------------------------------------------------------------#
# Q4.1.

# Displaying the KMS Curve
curves = Regression.curve_coordinates(y,1,0,predprob_event)
sen_spec = curves[["Threshold","Sensitivity","OneMinusSpecificity"]].copy()
melted = pd.melt(sen_spec,["Threshold"])
fig,axs = plt.subplots(figsize = (6,6))
print("\n######################################################################################\n")
lines = sns.lineplot(x="Threshold",y="value",hue="variable",data = melted,ax= axs)
plt.show()

# Calculating the Metrics
sen_spec["variation"] = sen_spec["Sensitivity"] - sen_spec["OneMinusSpecificity"]
kms_statistic = sen_spec["variation"].max()
kms_threshold = sen_spec[sen_spec["variation"] == sen_spec["variation"].max()]
kms_threshold = kms_threshold.iloc[0,0]
kms_value = Regression.binary_model_metric(target = y,valueEvent = 1,valueNonEvent = 0, predProbEvent = predprob_event,eventProbThreshold=kms_threshold)
miss_rate_kms = kms_value[2]

print("KMS Statistic = " + str(kms_statistic))
print("KMS Threshold = " + str(kms_threshold))
print("Missclassification Rate = " + str(miss_rate_kms))

#--------------------------------------------------------------------------------------------#
# Q4.2.

# Displaying the Precision Recall Curve with the no skill line
fig,axs = plt.subplots(figsize = (6,6))
print("\n######################################################################################\n")
lines_pre = sns.lineplot(data = curves,x="Recall",y="Precision",ax = axs)
no_skill_value = (len(y[y==1])/len(y))
plt.axhline(no_skill_value,0,1,linestyle="--")
plt.show()


f1 = curves["F1 Score"].max()
f1_threshold = curves[curves["F1 Score"] == f1].iloc[0,0]
f1_value = Regression.binary_model_metric(target = y,valueEvent = 1,valueNonEvent = 0, predProbEvent = predprob_event,eventProbThreshold=f1_threshold)
miss_rate_f1 = f1_value[2]

print("Max F1 = " + str(f1))
print("F1 Threshold = " + str(f1_threshold))
print("Missclassification Rate = " + str(miss_rate_f1))