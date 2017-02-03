"""
Hydropy package
@author: Stijn Van Hoey
from: https://github.com/stijnvanhoey/hydropy/tree/master/hydropy
for a better and more up to date copy of this script go to the original repo.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import numpy as np


def get_baseflow_chapman(flowserie, recession_time):
    """
    Parameters
    ----------
    flowserie :  pd.TimeSeries
        River discharge flowserie
    recession_time : float [0-1]
        recession constant
    Notes
    ------
    $$Q_b(i) = \frac{k}{2-k}Q_b(i-1) + \frac{1-k}{2-k}Q(i)$$
    """

    secterm = (1.-recession_time)*flowserie/(2.-recession_time)

    baseflow = np.empty(flowserie.shape[0])
    for i, timestep in enumerate(baseflow):
        if i == 0:
            baseflow[i] = 0.0
        else:
            baseflow[i] = recession_time*baseflow[i-1]/(2.-recession_time) + \
                            secterm.values[i]
    baseflow = pd.DataFrame(baseflow, index=flowserie.index)
    return baseflow


def get_baseflow_boughton(flowserie, recession_time, baseflow_index):
    """
    Parameters
    ----------
    flowserie :  pd.TimeSeries
        River discharge flowserie
    recession_time : float [0-1]
        recession constant
    baseflow_index : float
    Notes
    ------
    $$Q_b(i) = \frac{k}{1+C}Q_b(i-1) + \frac{C}{1+C}Q(i)$$
    """

    parC = baseflow_index

    secterm = parC*flowserie/(1 + parC)

    baseflow = np.empty(flowserie.shape[0])
    for i, timestep in enumerate(baseflow):
        if i == 0:
            baseflow[i] = 0.0
        else:
            baseflow[i] = recession_time*baseflow[i-1]/(1 + parC) + \
                            secterm.values[i]
    return pd.DataFrame(baseflow, index=flowserie.index)


def get_baseflow_ihacres(flowserie, recession_time, baseflow_index, alfa):
    """
    Parameters
    ----------
    flowserie :  pd.TimeSeries
        River discharge flowserie
    recession_time : float [0-1]
        recession constant
    Notes
    ------
    $$Q_b(i) = \frac{k}{1+C}Q_b(i-1) + \frac{C}{1+C}[Q(i)+\alpha Q(i-1)]$$
    $\alpha$ < 0.
    """

    parC = baseflow_index

    secterm = parC/(1 + parC)

    baseflow = np.empty(flowserie.shape[0])
    for i, timestep in enumerate(baseflow):
        if i == 0:
            baseflow[i] = 0.0
        else:
            baseflow[i] = recession_time * baseflow[i-1]/(1 + parC) + \
                            secterm * (flowserie.values[i] +
                                       alfa * flowserie.values[i-1])
    return pd.DataFrame(baseflow, index=flowserie.index)

def exp_curve(x, a, b):
    """Exponential curve used for rating curves"""
    return (a * x**b)

def ratingCurve(discharge, stage):
    """Computes rating curve based on discharge measurements coupled with stage
    readings.
    discharge = array of measured discharges;
    stage = array of corresponding stage readings;
    Returns coefficients a, b for the rating curve in the form y = a * x**b
    """
    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(exp_curve, stage, discharge)

    def r_squ():
        a = 0.0
        b = 0.0
        for i, j in zip(discharge, stage):
            a += (i - exp_curve(j, popt[0], popt[1]))**2
            b += (i - np.mean(discharge))**2
        return 1 - a / b

    return popt, r_squ()

def RB_Flashiness(series):
    """Richards-Baker Flashiness Index for a series of daily mean discharges.
    https://github.com/hydrogeog/hydro"""
    Qsum = np.sum(series)           # sum of daily mean discharges
    Qpath = 0.0
    for i in range(len(series)):
        if i == 0:
            Qpath = series[i]       # first entry only
        else:
            Qpath += np.abs(series[i] - series[i-1])    # sum the absolute differences of the mean discharges
    return Qpath/Qsum


def flow_duration(series):
    """Creates the flow duration curve for a discharge dataset. Returns a pandas
    series whose index is the discharge values and series is exceedance probability.
    https://github.com/hydrogeog/hydro"""
    fd = pd.Series(series).value_counts()               # frequency of unique values
    fd.sort_index(inplace=True)                         # sort in order of increasing discharges
    fd = fd.cumsum()                                    # cumulative sum of frequencies
    fd = fd.apply(lambda x: 100 - x/fd.max() * 100)     # normalize
    return fd

def Lyne_Hollick(series, alpha=.925, direction='f'):
    """Recursive digital filter for baseflow separation. Based on Lyne and Hollick, 1979.
    series = array of discharge measurements
    alpha = filter parameter
    direction = (f)orward or (r)everse calculation
    https://github.com/hydrogeog/hydro
    """
    series = np.array(series)
    f = np.zeros(len(series))
    if direction == 'f':
        for t in np.arange(1,len(series)):
            f[t] = alpha * f[t-1] + (1 + alpha)/2 * (series[t] - series[t-1])
            if series[t] - f[t] > series[t]:
                f[t] = 0
    elif direction == 'r':
        for t in np.arange(len(series)-2, 1, -1):
            f[t] = alpha * f[t+1] + (1 + alpha)/2 * (series[t] - series[t+1])
            if series[t] - f[t] > series[t]:
                f[t] = 0
    return np.array(series - f)

def Eckhardt(series, alpha=.98, BFI=.80):
    """Recursive digital filter for baseflow separation. Based on Eckhardt, 2004.
    series = array of discharge measurements
    alpha = filter parameter
    BFI = BFI_max (maximum baseflow index)
    https://github.com/hydrogeog/hydro
    """
    series = np.array(series)
    f = np.zeros(len(series))
    f[0] = series[0]
    for t in np.arange(1,len(series)):
        f[t] = ((1 - BFI) * alpha * f[t-1] + (1 - alpha) * BFI * series[t]) / (1 - alpha * BFI)
        if f[t] > series[t]:
            f[t] = series[t]
    return f
