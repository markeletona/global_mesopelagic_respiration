# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 08:48:15 2023

@author: Markel Gómez Letona

Python implementation of the 'RegressConsensus.m' and 'RegressConsensusW.m' 
Matlab routines by X.A. Alvarez-Salgado & Marta Alvarez from Alvarez-Salgado 
et al. (2014), doi:10.1016/j.pocean.2013.12.009

It differs from OLS simple linear regression (model I) in that it allows to 
account for errors in both the x and y variables*, giving a specific weight to 
each axis (provided by the user or as function the measurements uncertainties).
The specific case in which both axes are given the same weight (0.5) would be 
the same as a model II linear regression.

* model I puts all the error on the y axis only (assumes x variable is 
controlled/measured without error)
_______________________________________________________________________________

"""

from scipy.special import fdtr, stdtr
from scipy.stats import t as tdist
from numpy import sign, isnan, Inf, var

def RegressConsensusW(X, Y, Wx=.5, intercept=1):
    """
    Perform linear regression providing specific weights for each axis. Weights
    default to 0.5, i.e., a model II linear regression.

    Parameters
    ----------
    X : LIST
        Data of variable x.
    Y : LIST
        Data of variable y. Must be the same length as X.
    Wx : FLOAT (or INTEGER), optional
        Weigth of X, between [0,1]. Default is .5. For any Wx -> Wy = 1 - Wx
    intercept : INTEGER, optional
        Whether to force the intercept (a) to pass from 0. 
        Takes values of either 1 (a~=0) or 0 (a=0). The default is 1.

    Raises
    ------
    ValueError
        X or Y could not be coerced to list.
    ValueError
        X or Y must be flat lists.
    ValueError
        X and Y need to have at least 2 observations.
    ValueError
        X and Y need to have equal number of observations.
    ValueError
        Wx needs to be between [0,1].
    ValueError
        intercept needs to be either 1 (a~=0) or 0 (a=0).
        
    Returns
    -------
    DICT
        Dictionary with regression results:
            slope : slope of regression
            sse : standard error of slope
            sci95 : +/- value of the 95% confidence interval of slope
            intercept : intercept of regression
            ise : standard error of intercept
            ici95 : +/- value of the 95% confidence interval of intercept
            r2 : r squared (coefficient of determination)
            r2_adj : adjusted r2
            F : F statistic
            pvalue : p-value of the regression
            stvalue : t value of the slope
            itvalue : t value of the intercept
            spvalue : p-value of the slope
            ipvalue : p-value of the intercept
            n: no. of observations included in the regression

    """
    # # Hacky function to avoid needing to load modules to catch nans
    # def is_nan(num):
    #     return num != num
    # def sign_(a):
    #     return (a > 0) - (a < 0)
    
    # Check that X, Y are lists, if not try to coerce them:
    X = list(X)
    if (not isinstance(X, list)):
        raise ValueError('X could not be coerced to list')
    Y = list(Y)
    if (not isinstance(Y, list)):
        raise ValueError('Y could not be coerced to list')   
    
    # Check that X or Y are 'flat' lists (do not contain other lists):
    if any(isinstance(ix, list) for ix in X):
        raise ValueError('X must be a flat list (not have any sublists)')
    if any(isinstance(iy, list) for iy in Y):
        raise ValueError('Y must be a flat list (not have any sublists)')
     
    # Check that X,Y are the same length:
    if not len(X)==len(Y):
        raise ValueError('X and Y need to have equal number of observations')
        
    # Check that Wx is a float (try to coerce if not) between 0 and 1:
    if (not isinstance(Wx, float)):
        Wx = float(Wx)
    if (Wx>1) or (Wx<0):
        raise ValueError('Wx needs to be between [0,1]')
    
    # Check that intercept is either 1 or 0:
    if not ((intercept==1) or (intercept==0)):
        raise ValueError('intercept needs to be either 1 (a~=0) or 0 (a=0)')
        
        
    # Remove any value pairs with NaN in them:
    # (find nans in X,Y and then retain pairs in which BOTH have valid values)
    X_isnan = [isnan(i) for i in X]
    Y_isnan = [isnan(j) for j in Y]
    boo = [(not i) and (not j) for i, j in zip(X_isnan, Y_isnan)]
    X = [v for (v, b) in zip(X, boo) if b]
    Y = [v for (v, b) in zip(Y, boo) if b]
    
    # Determine the length of vectors
    n = len(X)
    
    # Check that X,Y have at least two observations:
    if n<2:
        raise ValueError('X and Y need to have at least 2 observations')
    
    # Determine residual degrees of freedom
    # nu = n-k, k is 1 for the slope, we are dealing with one X variable
    nu = n - 1 
    if intercept==1:
        nu = nu - 1 # subtract one more if intercept is also estimated
    
    # Calculate the required sums for the formulas
    Sx = sum(X)
    Sy = sum(Y)
    Sx2 = sum([i ** 2 for i in X]) # sum(X^2)
    Sy2 = sum([j ** 2 for j in Y]) # sum(Y^2)
    Sxy = sum([i * j for i, j in zip(X,Y)]) # sum(X*Y)
    
    # Also means:
    mx = sum(X)/len(X)
    my = sum(Y)/len(Y)
    
    # Calculate correlation coefficient:
    num = n * Sxy - Sx * Sy
    den = n * Sx2 - Sx ** 2
    R = num / ((den ** .5) * ((n * Sy2 - Sy ** 2) ** .5));
   
    # Calculate the Y weight
    Wy = 1 - Wx

    # IF the intercept equals 0 (a=0)
    if intercept==0:
        a=0
        if Wx==0:
            b = Sxy / Sx2
        elif Wx==1:
            b = sign(R) * Sy2 / Sxy
        else:
            b = ((1 - 2 * Wx) / (2 - 2 * Wx) * (Sxy / Sx2) +
                 sign(R) * (((1 - 2 * Wx) * Sxy) ** 2 +
                             4 * Wx * Wy * (Sx2 * Sy2)) ** .5 / 
                 (2 * Wy * Sx2))


    # IF the intercept is different from 0 (a~=0)
    elif intercept==1:
        if Wx==0:
            b = (n * Sxy - Sx * Sy) / (n * Sx2 - Sx ** 2)
        elif Wx==1:
            b = (n * Sy2 - Sy ** 2) / (n * Sxy - Sx * Sy)
        elif Wx==.5:
            b = sign(R) * ((n * Sy2 - sum(Y) ** 2) /
                            (n * Sx2 - Sx ** 2)) ** .5
        else:
            b = (((1 - 2 * Wx) / (2 * Wy)) * (n * Sxy - Sx * Sy) /
                 (n * Sx2 - Sx ** 2) +
                 (sign(R) * (((2 * Wx - 1) * (n * Sxy - Sx * Sy)) ** 2 +
                              4 * Wx * Wy * (n * Sx2 - Sx ** 2) *
                              (n * Sy2 - Sy**2)) ** .5 /
                  (2 * Wy * (n * Sx2 - Sx ** 2))))
      
        a = my - b * mx


    # Predicted values
    yhat = [a + b * x for x in X] # a + b * X
    
    # Residuals
    r = [y - yh for y, yh in zip(Y, yhat)] # Y - yhat
    
    # Root mean square error
    if not nu==0:
        rmse = (sum([v ** 2 for v in r]) ** .5)/(nu ** .5) # norm(r)/sqrt(nu)
    else:
        rmse = float("inf")
        
    # Estimator of error variance
    s2 = rmse ** 2
    
    ## Calculate R-squared, F & p-value.
    # 
    # Regression sum of squares
    yhat_minus_ymean = [(yh - my) for yh in yhat] # yhat-mean(Y)
    RSS = (sum([v ** 2 for v in yhat_minus_ymean]) ** .5) ** 2  # norm(yhat-mean(Y))^2
    
    # R-square statistic
    r2 = R ** 2
    k = 1
    r2_adj = 1 - (1 - r2) * ((n - 1) / (n - k - 1)) # n = no. of obs.; k = no. of variables
    if intercept==1:
        # F statistic for regression
        if s2==0: # s2 can be 0 for perfect fits and Python cannot divide by 0
            F = Inf
        else:
            F = (RSS / (n - nu - 1)) / s2 
        # Significance probability for regression
        pval = 1 - fdtr(n - nu - 1, nu, F)
    else:
        if s2==0:
            F = Inf
        else:
            F = (RSS / (n - nu)) / s2
        pval = 1 - fdtr(n - nu, nu, F)
    
    # Standard error of the slope and intercept
    s2 = sum([i * i for i in r]) / nu # sum(r * r) / nu
    sb = (n * s2 / den) ** .5
    if intercept==1: # YES intercept
        sa = (Sx2 * s2 / den) ** .5
    else:
        sa = 0

    # Significance probability for slope and intercept
    if intercept==1:
        # t values, tb = b/sb 
        # (perfect fits yield sb, sa = 0, avoid warning in 0 division)
        tb = (b/sb if not sb==0 else Inf) # tb = b/sb
        ta = (a/sa if not sa==0 else Inf) # ta = a/sa
        # p-values:
        pval_b = 2 * (1 - stdtr(nu, abs(tb))) # pval and pval_b should be the same
        pval_a = 2 * (1 - stdtr(nu, abs(ta)))
        # 95% confidence intervals (the +/- value)
        ci_b = tdist.ppf((1 - .05 / 2), nu) * sb
        ci_a = tdist.ppf((1 - .05 / 2), nu) * sa
    else:
        tb = (b/sb if not sb==0 else Inf) # tb = b/sb
        ta = Inf # for intercept==0, a=0, sa=0
        pval_b = 2 * (1 - stdtr(nu, abs(tb)))
        pval_a = 2 * (1 - stdtr(nu, abs(ta)))
        ci_b = tdist.ppf((1 - .05 / 2), nu) * sb
        ci_a = 0

    # Gather results:
    out = {'slope': b,
           'sse': sb,
           'sci95': ci_b,
           'intercept': a,
           'ise': sa,
           'ici95': ci_a,
           'r2': r2,
           'r2_adj': r2_adj,
           'F': F,
           'pvalue': pval,
           'stvalue': tb,
           'itvalue': ta,
           'spvalue': pval_b,
           'ipvalue': pval_a,
           'n': n}
    return(out)


def RegressConsensus(X, Y, sX, sY, intercept=1):
    """
    Perform linear regression estimating weights for each axis based on the
    uncertainties of the X and Y variables. 

    Parameters
    ----------
    X : LIST
        Data of variable x.
    Y : LIST
        Data of variable y. Must be the same length as X.
    sX : FLOAT
        Uncertainty or error of X.
    sY : FLOAT
        Uncertainty or error of Y.
    intercept : INTEGER, optional
        Whether to force the intercept (a) to pass from 0. 
        Takes values of either 1 (a~=0) or 0 (a=0). The default is 1.

    Raises
    ------
    ValueError
        X or Y could not be coerced to list.
    ValueError
        X or Y must be flat lists.
    ValueError
        X and Y need to have at least 2 observations.
    ValueError
        X and Y need to have equal number of observations.
    ValueError
        intercept needs to be either 1 (a~=0) or 0 (a=0).
        
    Returns
    -------
    DICT
        Dictionary with regression results:
            slope : slope of regression
            sse : standard error of slope
            sci95 : +/- value of the 95% confidence interval of slope
            intercept : intercept of regression
            ise : standard error of intercept
            ici95 : +/- value of the 95% confidence interval of intercept
            r2 : r squared (coefficient of determination)
            r2_adj : adjusted r2
            F : F statistic
            pvalue : p-value of the regression
            stvalue : t value of the slope
            itvalue : t value of the intercept
            spvalue : p-value of the slope
            ipvalue : p-value of the intercept
            n: no. of observations included in the regression

    """
    # # Hacky function to avoid needing to load modules to catch nans
    # def is_nan(num):
    #     return num != num
    # def sign_(a):
    #     return (a > 0) - (a < 0)
    
    # Check that X, Y are lists, if not try to coerce them:
    X = list(X)
    if (not isinstance(X, list)):
        raise ValueError('X could not be coerced to list')
    Y = list(Y)
    if (not isinstance(Y, list)):
        raise ValueError('Y could not be coerced to list')   
    
    # Check that X or Y are 'flat' lists (do not contain other lists):
    if any(isinstance(ix, list) for ix in X):
        raise ValueError('X must be a flat list (not have any sublists)')
    if any(isinstance(iy, list) for iy in Y):
        raise ValueError('Y must be a flat list (not have any sublists)')
    
    # Check that X,Y have at least two observations:
    if (len(X)<2) or (len(Y)<2):
        raise ValueError('X and Y need to have at least 2 observations')
        
    # Check that X,Y are the same length:
    if not len(X)==len(Y):
        raise ValueError('X and Y need to have equal number of observations')
    
    # Try to coerce sX and sY to floats:
    if (not isinstance(sX, float)):
        sX = float(sX)
    if (not isinstance(sY, float)):
        sY = float(sY)

    # Check that intercept is either 1 or 0:
    if not ((intercept==1) or (intercept==0)):
        raise ValueError('intercept needs to be either 1 (a~=0) or 0 (a=0)')
        
        
    # Remove any value pairs with NaN in them:
    # (find nans in X,Y and then retain pairs in which BOTH have valid values)
    X_isnan = [isnan(i) for i in X]
    Y_isnan = [isnan(j) for j in Y]
    boo = [(not i) and (not j) for i, j in zip(X_isnan, Y_isnan)] 
    X = [v for (v, b) in zip(X, boo) if b]
    Y = [v for (v, b) in zip(Y, boo) if b]
    
    # Determine the length of vectors
    n = len(X)
    
    # Determine residual degrees of freedom
    # nu = n-k, k is 1 for the slope, we are dealing with one X variable
    nu = n - 1 
    if intercept==1:
        nu = nu - 1 # subtract one more if intercept is also estimated
    
    # Calculate the required sums for the formulas
    Sx = sum(X)
    Sy = sum(Y)
    Sx2 = sum([i ** 2 for i in X]) # sum(X^2)
    Sy2 = sum([j ** 2 for j in Y]) # sum(Y^2)
    Sxy = sum([i * j for i, j in zip(X,Y)]) # sum(X*Y)
    
    # Also means:
    mx = sum(X)/len(X)
    my = sum(Y)/len(Y)
    
    # Calculate correlation coefficient:
    num = n * Sxy - Sx * Sy
    den = n * Sx2 - Sx ** 2
    R = num / ((den ** .5) * ((n * Sy2 - Sy ** 2) ** .5));
   
    # Calculate the weighting factors:
    Wx = (sX ** 2 / var(X)) / (sX ** 2 / var(X) + sY ** 2 / var(Y))
    Wy = 1 - Wx

    # IF the intercept equals 0 (a=0)
    if intercept==0:
        a=0
        if Wx==0:
            b = Sxy / Sx2
        elif Wx==1:
            b = sign(R) * Sy2 / Sxy
        else:
            b = ((1 - 2 * Wx) / (2 - 2 * Wx) * (Sxy / Sx2) +
                 sign(R) * (((1 - 2 * Wx) * Sxy) ** 2 +
                             4 * Wx * Wy * (Sx2 * Sy2)) ** .5 / 
                 (2 * Wy * Sx2))


    # IF the intercept is different from 0 (a~=0)
    elif intercept==1:
        if Wx==0:
            b = (n * Sxy - Sx * Sy) / (n * Sx2 - Sx ** 2)
        elif Wx==1:
            b = (n * Sy2 - Sy ** 2) / (n * Sxy - Sx * Sy)
        elif Wx==.5:
            b = sign(R) * ((n * Sy2 - sum(Y) ** 2) /
                            (n * Sx2 - Sx ** 2)) ** .5
        else:
            b = (((1 - 2 * Wx) / (2 * Wy)) * (n * Sxy - Sx * Sy) /
                 (n * Sx2 - Sx ** 2) +
                 (sign(R) * (((2 * Wx - 1) * (n * Sxy - Sx * Sy)) ** 2 +
                              4 * Wx * Wy * (n * Sx2 - Sx ** 2) *
                              (n * Sy2 - Sy**2)) ** .5 /
                  (2 * Wy * (n * Sx2 - Sx ** 2))))
      
        a = my - b * mx


    # Predicted values
    yhat = [a + b * x for x in X] # a + b * X
    
    # Residuals
    r = [y - yh for y, yh in zip(Y, yhat)] # Y - yhat
    
    # Root mean square error
    if not nu==0:
        rmse = (sum([v ** 2 for v in r]) ** .5)/(nu ** .5) # norm(r)/sqrt(nu)
    else:
        rmse = float("inf")
        
    # Estimator of error variance
    s2 = rmse ** 2
    
    ## Calculate R-squared, F & p-value.
    # 
    # Regression sum of squares
    yhat_minus_ymean = [(yh - my) for yh in yhat] # yhat-mean(Y)
    RSS = (sum([v ** 2 for v in yhat_minus_ymean]) ** .5) ** 2  # norm(yhat-mean(Y))^2
    
    # R-square statistic
    r2 = R ** 2
    k = 1
    r2_adj = 1 - (1 - r2) * ((n - 1) / (n - k - 1)) # n = no. of obs.; k = no. of variables
    if intercept==1:
        # F statistic for regression
        if s2==0: # s2 can be 0 for perfect fits and Python cannot divide by 0
            F = Inf
        else:
            F = (RSS / (n - nu - 1)) / s2 
        # Significance probability for regression
        pval = 1 - fdtr(n - nu - 1, nu, F)
    else:
        if s2==0:
            F = Inf
        else:
            F = (RSS / (n - nu)) / s2
        pval = 1 - fdtr(n - nu, nu, F)
   
    # Standard error of the slope and intercept
    s2 = sum([i * i for i in r]) / nu # sum(r * r) / nu
    sb = (n * s2 / den) ** .5
    if intercept==1: # YES intercept
        sa = (Sx2 * s2 / den) ** .5
    else:
        sa = 0
    
    # Significance probability for slope and intercept
    if intercept==1:
        # t values
        # (perfect fits yield sb, sa = 0, avoid warning in 0 division)
        tb = (b/sb if not sb==0 else Inf) # tb = b/sb
        ta = (a/sa if not sa==0 else Inf) # ta = a/sa
        # p-values:
        pval_b = 2 * (1 - stdtr(nu, tb)) # pval and pval_b should be the same
        pval_a = 2 * (1 - stdtr(nu, ta))
        # 95% confidence intervals (the +/- value)
        ci_b = tdist.ppf((1 - .05 / 2), nu) * sb
        ci_a = tdist.ppf((1 - .05 / 2), nu) * sa
    else:
        tb = (b/sb if not sb==0 else Inf) # tb = b/sb
        ta = Inf # for intercept==0, a=0, sa=0
        pval_b = 2 * (1 - stdtr(nu, tb))
        pval_a = 2 * (1 - stdtr(nu, ta))
        ci_b = tdist.ppf((1 - .05 / 2), nu) * sb
        ci_a = 0


    # Gather results:
    out = {'slope': b,
           'sse': sb,
           'sci95': ci_b,
           'intercept': a,
           'ise': sa,
           'ici95': ci_a,
           'r2': r2,
           'r2_adj': r2_adj,
           'F': F,
           'pvalue': pval,
           'stvalue': tb,
           'itvalue': ta,
           'spvalue': pval_b,
           'ipvalue': pval_a,
           'n': n}
    return(out)

if __name__ == '__main__':
    # Create example data
    X = [1, 3, 4, 11, 18, 20, 21, 24, 32, 37, 40, 41, 47, 50, 59, 62, 
         67, 73, 78, 79, 85, 89, 94, 99]
    Y = [41, 54, 91, 70, 127, 282, 149, 259, 316, 240, 348, 299, 371, 432,
         393, 419, 483, 481, 619, 549, 644, 553, 549, 589]
    sX, sY = [2.4, 10.3]
    # ------------
    print("Running examples ...")
    print("RegressConsensus: ")
    RegressConsensus(X, Y, sX, sY, intercept=1)
    print("RegressConsensusW: ")
    RegressConsensusW(X, Y, Wx=0, intercept=0)
    