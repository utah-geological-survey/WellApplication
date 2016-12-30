# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 19:55:22 2016

@author: p
"""

import numpy as np
import pandas as pd
import scipy as sc
from scipy.stats import norm


def mk_test(x, alpha = 0.05):
    """This perform the MK (Mann-Kendall) test to check if there is any trend present in 
    data or not
    
    Args:
        x:   a vector of data
        alpha: significance level
    
    Returns:
        trend: tells the trend (increasing, decreasing or no trend)
        h: True (if trend is present) or False (if trend is absence)
        p: p value of the sifnificance test
        z: normalized test statistics 
        
    Examples::
      >>> x = np.random.rand(100)
      >>> trend = mk_test(x,0.05)
      >>> print(trend.trend)
      increasing
      
    Credit: http://pydoc.net/Python/ambhas/0.4.0/ambhas.stats/
    """
    n = len(x)
    ta = n*(n-1)/2
    # calculate S 
    s = 0
    for k in xrange(n-1):
        for j in xrange(k+1,n):
            s += np.sign(x[j] - x[k])
    
    # calculate the unique data
    unique_x = np.unique(x)
    g = len(unique_x)
    
    # calculate the var(s)
    if n == g: # there is no tie
        var_s = (n*(n-1)*(2*n+5))/18
    else: # there are some ties in data
        tp = np.zeros(unique_x.shape)
        for i in xrange(len(unique_x)):
            tp[i] = sum(unique_x[i] == x)
        var_s = (n*(n-1)*(2*n+5) - np.sum(tp*(tp-1)*(2*tp+5)))/18
    
    if s>0:
        z = (s - 1)/np.sqrt(var_s)
    elif s == 0:
        z = 0
    elif s<0:
        z = (s + 1)/np.sqrt(var_s)
    else:
        z = 0
    
    # calculate the p_value
    p = 2*(1-sc.stats.norm.cdf(abs(z))) # two tail test
    h = abs(z) > sc.stats.norm.ppf(1-alpha/2) 
    
    if (z<0) and h:
        trend = 'decreasing'
    elif (z>0) and h:
        trend = 'increasing'
    else:
        trend = 'no trend'
        
    return pd.Series({'trend':trend, 'varS':round(var_s,3), 'p':round(p,3), 'z':round(z,3), 's':round(s,3), 'n':n, 'ta':ta})
    
def mk_ts(df, const, group1, orderby = 'year', alpha = 0.05):
    '''
    df = dataframe
    const = variable tested for trend
    group1 = variable to group by
    orderby = variable to order by (typically a date)
    '''
    
    def zcalc(Sp, Varp):
        if Sp > 0:
            return (Sp - 1)/Varp**0.5
        elif Sp < 0:
            return (Sp + 1)/Varp**0.5
        else:
            return 0   
    
    df.is_copy = False
    
    df[const] = df.ix[:,const].convert_objects(convert_numeric=True)
    # remove null values
    df[const].dropna(inplace=True)
    # remove index
    df.reset_index(inplace=True, drop=True)
    # sort by groups, then time
    df.sort(columns=[group1,orderby],axis=0, inplace=True)
    
    # group by group and apply mk_test
    dg = df.groupby(group1).apply(lambda x: mk_test(x.loc[:,const].dropna().values, alpha))
    Var_S = dg.loc[:,'varS'].sum()
    S = dg.loc[:,'s'].sum()
    N = dg.loc[:,'n'].sum()
    Z = zcalc(S,Var_S)
    P = 2*(1-norm.cdf(abs(Z)))
    group_n = len(dg)
    h = abs(Z) > norm.ppf(1-alpha/2) 
    tau = S/dg.loc[:,'ta'].sum()

    if (Z<0) and h:
        trend = 'decreasing'
    elif (Z>0) and h:
        trend = 'increasing'
    else:
        trend = 'no trend'
    
    
    return pd.Series({'S':S, 'Z':round(Z,2), 'p':P, 'trend':trend, 'group_n':group_n, 'sample_n':N, 'Var_S':Var_S, 'tau':round(tau,2)})

    
