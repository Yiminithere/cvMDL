import numpy as np
import scipy.optimize as spo
from scipy.interpolate import interp1d

"""
This module contains the necessary functions used in the MFDistributionLearning class

"""

## compute powerset of {0, ..., n-1}
def subsets(n):
    if n == 0:
        return [[]]
    else:
        x = subsets(n-1)
        return x + [y + [n-1] for y in x]

## compute the set of subsets of {0, ..., n-1} with sizes <= m
def get_model_list(n, m):
    x = subsets(n)
    res = [y for y in x if 0<len(y)<=m]
    return res 

## get cost of each subset of c
def cost_match(c):
    c = np.array(c)
    len_c = len(c)
    subset_list = get_model_list(len_c, len_c)
    res = [np.sum(c[S]) for S in subset_list]
    return res

## estimate the J0 function of the empirical CDF of x (F_x): int_{R} F_x(z)(1-F_x(z)) dz
def J0_eval(x):
    x.sort()
    n = len(x)
    integrand_val = [k/n*(1-k/n) for k in range(1,n)]
    return sum([integrand_val[k]*(x[k+1]-x[k]) for k in range(n-1)])  

## estimate the J1 function of the empirical CDF of x (F_x): int_{R} sqrt(F_x(z)(1-F_x(z))) dz
def J1_eval(x):
    x.sort()
    n = len(x)
    # compute the integrand of J1
    intgrd_val = [np.sqrt(k/n)*np.sqrt(1-k/n) for k in range(1,n)]
    res = sum([intgrd_val[k]*(x[k+1]-x[k]) for k in range(n-1)])
    return res

## add intercept column to the design matrix df
def add_intercept(df):
    m = df.shape[0]
    if len(df.shape) == 1:
        df = df.reshape((m, 1))
    a = np.array([1]*m).reshape((m,1))
    return np.concatenate((a,df), axis=1)

## solve least squares AX = Y
def model_ls(A, Y):
    if len(Y.shape) == 1:
        Y = Y.reshape(len(Y), 1)
    X = np.linalg.lstsq(A, Y, rcond=None)[0]
    fit = A@X
    residuals = Y - fit
    sd = np.std(residuals, axis = 0)
    res = {'Beta': X, 'fit': fit,\
           'residuals': residuals, 'sd': sd}
    return res

## compute the integral of f with respect to the empirical CDF of x union y
def integration(x, y, f):
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    x_y = np.sort(np.concatenate((x, y)))
    point_val = np.array([f(x_y[i])*(x_y[i+1]-x_y[i]) for i in range(len(x_y)-1)])
    return np.sum(point_val, axis = 0)    

## compute the inverse of a CDF f
def cdf_inverse(f, interpolation = False, interval = []):
    lowerbound = -1.0e10
    upperbound = 1.0e10
    if interpolation == False:
        def f_inverse(alpha):
            if f(lowerbound)>=alpha:
                return -np.inf
            elif f(upperbound)<=alpha:
                return np.inf
            else:
                def g(x):
                    return f(x) - alpha
                init_up = 1
                init_low = -1
                while f(init_up)<=alpha:
                    init_up *= 2
                while f(init_low)>=alpha:
                    init_low *= 2
                result = spo.root_scalar(g, bracket=[init_low, init_up])
                return result.root
    else:
        N_interp = 1000
        x_min, x_max = interval
        x = np.linspace(x_min, x_max, num=N_interp)
        y = np.array([f(z) for z in x])
        inv_func = interp1d(y, x)
        def f_inverse(alpha):
            if alpha<=0:
                return f(x_min)
            elif alpha>=1:
                return f(x_max)
            else:
                return inv_func(alpha)     
    return f_inverse

## sort a 2d distribution
def sort_2d(A):
    itermax = 1000
    A = np.array(A)
    if len(A.shape)!=2:
        raise TypeError('Incorrect dimension.')
    else:
        m, n = A.shape
        stop = False
        N = 0
        while stop == False & N<=itermax:
            A_temp = A.copy()
            for i in range(m):
                A[i,:] = np.sort(A[i,:])
            for j in range(n):
                A[:,j] = np.sort(A[:,j])
            if np.max(A_temp-A)==0:
                stop = True
            N += 1
    return A