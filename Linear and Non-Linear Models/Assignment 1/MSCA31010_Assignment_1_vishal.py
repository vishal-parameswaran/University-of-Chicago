#Initializations
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tkinter import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import math
import sys
sns.set_theme(style="ticks")
np.set_printoptions(precision = 10, threshold = sys.maxsize)
np.set_printoptions(linewidth = np.inf)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)
pd.options.display.float_format = '{:,.10}'.format

#Read the Data
data = pd.read_csv("Economy_2020_to_2022.csv")
print(data.head(5))

#Question 1.1 Show the Scatter plot matrix
sns.set_style("ticks",{'axes.grid' : True})
fig = sns.pairplot(data[["PCEPI","CPIAUCSL","ICSA_Week1","ICSA_Week2","ICSA_Week3","ICSA_Week4","ICSA_Week5"]],diag_kind="none")
plt.show()

sns.set()
# The image will take some time to show. Please remember to close it, to be able to see the remaining outputs.

column_headers = np.array(["PCEPI","CPIAUCSL","ICSA_Week1","ICSA_Week2","ICSA_Week3","ICSA_Week4","ICSA_Week5"])

#Question 1.2 Compute the Pearson Correlation Co-Efficents
pearson_correlation_coeffs = data[column_headers].corr(method='pearson')
pearson_correlation_coeffs = pearson_correlation_coeffs.round(4)
print(pearson_correlation_coeffs)

#Question 1.3 Compute the Spearman Correlation Co-Efficents
spearman_correlation_coeffs = data[column_headers].corr(method='spearman')
spearman_correlation_coeffs = spearman_correlation_coeffs.round(4)
print(spearman_correlation_coeffs)

#Question 1.4 Compute the Kendall Correlation Co-Efficents
kendall_correlation_coeffs = data[column_headers].corr(method='kendall')
kendall_correlation_coeffs = kendall_correlation_coeffs.round(4)
print(kendall_correlation_coeffs)

# A function to calculate the Distance Correlation Coefficient between two columns
def distance_correlation (columns):
    #Calculate the Adjusted Distance for x

    column_x = columns.split(";")[0]
    column_y = columns.split(";")[1]
    dropped_rows = data[[column_x,column_y]].dropna()
    x = dropped_rows[[column_x]]
    y = dropped_rows[[column_y]]
    x = x.fillna(0)
    i = x.to_numpy().T[0]
    n = len(i)
    distance_matrix_x = pd.DataFrame(np.abs(i[:, None] - i))
    distance_matrix_x = distance_matrix_x.sub(distance_matrix_x.mean(axis=0),axis=1)
    distance_matrix_x = distance_matrix_x.sub(distance_matrix_x.mean(axis=1),axis=0)
    emp_dist_x = distance_matrix_x.pow(2)
    emp_dist_x = emp_dist_x.values.sum()/(n**2)
    #Calculate the Adjusted Distance for y
    y = y.fillna(0)
    d = y.to_numpy().T[0]
    n = len(i)
    distance_matrix_y = pd.DataFrame(np.abs(d[:, None] - d))
    distance_matrix_y = distance_matrix_y.sub(distance_matrix_y.mean(axis=0),axis=1)
    distance_matrix_y = distance_matrix_y.sub(distance_matrix_y.mean(axis=1),axis=0)
    emp_dist_y = distance_matrix_y.pow(2)
    emp_dist_y = emp_dist_y.values.sum()/(n**2)
    #Calculate the Distance covariance
    emp_dist_xy = distance_matrix_x.mul(distance_matrix_y,axis="index")
    emp_dist_xy = emp_dist_xy.values.sum()/(n**2)
    #Calculate the Distance Correlation Coefficient
    distance_correlation_coeff = math.sqrt((emp_dist_xy)/(math.sqrt(emp_dist_x*emp_dist_y)))
    return round(distance_correlation_coeff,4)

#Question 1.5 Compute the Distance Correlation Co-Efficents
distance_corr_matrix = pd.DataFrame(np.core.defchararray.add(column_headers[:, None],np.char.add(";",column_headers)),index=["PCEPI","CPIAUCSL","ICSA_Week1","ICSA_Week2","ICSA_Week3","ICSA_Week4","ICSA_Week5"],columns = ["PCEPI","CPIAUCSL","ICSA_Week1","ICSA_Week2","ICSA_Week3","ICSA_Week4","ICSA_Week5"])
distance_corr_matrix = distance_corr_matrix.applymap(distance_correlation)
print(distance_corr_matrix)

#A Group of Function to calculate the Newton Raphson methods.
def func (x,a):
    y = x**2 - a
    return (y)

def dfunc(x):
    dy = 2*x
    return (dy)

def newton_raphson (a,init_x, max_iter = 100, eps_conv = 1e-7, q_history = False,):
    x_value = pd.Series(np.arange(-10.0, 8.1, 0.1), name = 'x')
    y_value = x_value.apply(func,args=[a])
    y_value.name = 'y'
    i_iter = 0
    q_continue = True
    reason = 0
    x_curr = init_x
    if (q_history):
        history = []
    while (q_continue):
        f_curr = func(x_curr,a)
        dfunc_curr = dfunc(x_curr)
        if (q_history):
            history.append([i_iter, x_curr, f_curr, dfunc_curr])
        if (f_curr != 0.0):
            if (dfunc_curr != 0.0):
                i_iter = i_iter + 1
                x_next = x_curr - f_curr / dfunc_curr
                if (abs(x_next - x_curr) <= eps_conv):
                    q_continue = False
                    reason = 1               # Successful convergence
                elif (i_iter >= max_iter):
                    q_continue = False
                    reason = 2               # Exceeded maximum number of iterations
                else:
                    x_curr = x_next
            else:
                q_continue = False
                reason = 3                  # Zero derivative
        else:
            q_continue = False
            reason = 4                     # Zero function value

    if(q_history):
        print(pd.DataFrame(history, columns = ['Iteration', 'Estimate', 'Function', 'Derivative']))

    return (x_curr, reason)

#Question 2.3
x_solution, reason = newton_raphson (9,init_x = 1, max_iter = 10000, eps_conv = 1e-13, q_history = True)
#Question 2.4
x_solution, reason = newton_raphson (9000,init_x = 1, max_iter = 10000, eps_conv = 1e-13, q_history = True )
#Question 2.5
x_solution, reason = newton_raphson (0.0000009,init_x = 1, max_iter = 10000, eps_conv = 1e-13, q_history = True)
