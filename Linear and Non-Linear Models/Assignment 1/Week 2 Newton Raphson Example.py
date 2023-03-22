# -*- coding: utf-8 -*-
"""
@name: Week 2 Newton Raphson Example.py
@author: Ming-Long Lam, Ph.D.
@organization: University of Chicago
"""
import matplotlib.pyplot as plt
import numpy
import pandas
import sys

# Set some options for printing all the columns
numpy.set_printoptions(precision = 10, threshold = sys.maxsize)
numpy.set_printoptions(linewidth = numpy.inf)

pandas.set_option('display.max_columns', None)
pandas.set_option('display.expand_frame_repr', False)
pandas.set_option('max_colwidth', None)

pandas.options.display.float_format = '{:,.10}'.format

def func (x):
   y = x**2 - 9
   return (y)

def dfunc(x):
   dy = 2*x
   return (dy)

def newton_raphson (init_x, max_iter = 100, eps_conv = 1e-7, q_history = False):
   i_iter = 0
   q_continue = True
   reason = 0
   x_curr = init_x

   if (q_history):
      history = []
   while (q_continue):
      f_curr = func(x_curr)
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
      print(pandas.DataFrame(history, columns = ['Iteration', 'Estimate', 'Function', 'Derivative']))
          
   return (x_curr, reason)

x_value = pandas.Series(numpy.arange(-10.0, 8.1, 0.1), name = 'x')
y_value = x_value.apply(func)
y_value.name = 'y'

plt.figure(figsize = (10,6), dpi = 200)
plt.plot(x_value, y_value)
plt.title('y = x^4 + 4 * x^3 − 34 * x^2 − 76 * x + 105')
plt.xlabel('x')
plt.ylabel('y')
plt.xticks(range(-10,9,1))
plt.yticks(range(-500,4500,500))
plt.grid()
plt.show()   

x_solution, reason = newton_raphson (init_x = -10, max_iter = 100, eps_conv = 1e-7, q_history = True)

x_solution, reason = newton_raphson (init_x = -2, max_iter = 100, eps_conv = 1e-7, q_history = True)

x_solution, reason = newton_raphson (init_x = 0, max_iter = 100, eps_conv = 1e-7, q_history = True)

x_solution, reason = newton_raphson (init_x = 8, max_iter = 100, eps_conv = 1e-7, q_history = True)