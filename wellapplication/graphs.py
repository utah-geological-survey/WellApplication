# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 12:32:51 2015

@author: paulinkenbrandt
"""
from scipy import stats as sp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from datetime import datetime, timedelta
from scipy.optimize import curve_fit


def get_recess_int(df, Q, maxper=18, minper=6, thresh=30, inplace=False):
    """Gets intervals of recession from a hydrograph

    :param df: DataFrame with hydrograph data
    :param Q: Field in DataFrame with discharge data
    :param maxper: Period of record to scan discharge data for maxima; created for daily values; defaults to 18
    :param minper: Period of record to scan discharge data for minima; should be less than maxper; defaulst to 6
    :param thresh: Threshold of discharge below which maxima are not considered; defaults to 30
    :param inplace: Append to input database or create new one; defaults to False
    :return: DataFrame of original data and Max and Min, Indexes of maxima, Indexes of minima
    """
    if inplace:
        data = df
    else:
        data = df.copy()
    data['max'] = data[Q].rolling(maxper,center=True).max()
    data['max'] = data.ix[data['max'] == data['value'],'max']
    data['max'] = data.ix[data['max'] > thresh, 'max']
    data['min'] = data[Q].rolling(minper,center=True).min()
    data['min'] = data.ix[data['min'] == data['value'],'min']

    maxlist = data.index[data['max'].notnull()]
    firstmin = []
    for ind in maxlist:
        firstmin.append(data.ix[ind:,'min'].first_valid_index())
    data['min'] = data.ix[data.index.isin(firstmin),'min']
    return data, maxlist, firstmin


class recess(object):
    """Creates recession curve and modeled output to describe spring and streamflow recession.

    :param df: dataframe with spring discharge data
    :param Q: string indicating discharge field in df in units of gpm
    :param st: start date to examine data in [YYYY, MM, DD] format, where values are integers in an array
    :param end: end date to examine data
    :param excs: = begin date of exclusion period
    :param excf: = end date of exclusion period

    :type st: list
    :type end: list
    :type Q: str

    :return popt: alpha value for recession curve
    :return x1: days from start of recession
    :return x2: dates of recession curve analysis
    :return y1: points used for recession curve analysis
    :return y2: recession curve values
    :returns: Plot of recession curve
    """

    def __init__(self, df, Q, st, end='', lab='', excs=[0, 0, 0], excf=[0, 0, 0]):
        self.ymd = [datetime.now().year, datetime.now().month, datetime.now().day]
        if end == '':
            end = self.ymd

        if lab == '':
            self.Qlab = 'Discharge'
        else:
            self.Qlab = lab

        self.Qz = df[Q][0]

        self.rec_results = self.recession(df, Q, st, end, excs, excf)

    def fitit(self, x, y, Q):
        from scipy.optimize import curve_fit

        func = lambda x, c: Q * np.exp(-1 * c * x)

        popt, pcov = curve_fit(func, x, y, p0=(1e-1))
        return popt, pcov


    def recession(self, df, Q, st, end, excs, excf):
        """Creates recession curve and modeled output to describe spring and streamflow recession.

        The user puts in a dataframe with discharge data and defines the date range of recession.  The Class will return
        recession values of the date range given.

        :param df: DataFrame with spring discharge data
        :type df: pandas.core.frame.DataFrame
        :param Q: discharge field in df in units of gpm
        :type Q: str
        :param st: start date to examine data in [YYYY, MM, DD] format, where values are integers in an array
        :type st: list
        :param end: end date to examine data
        :type end: list
        :param excs: begin date of exclusion period
        :param excf: end date of exclusion period
        :type excs: list
        :type excs: list

        :returns: popt1, x1, x2, y1, y2
        :return popt1: alpha value for recession curve
        :return x1: days from start of recession
        :return x2: dates of recession curve analysis
        :return y1: points used for recession curve analysis
        :return y2: recession curve values
        """
        # account for hours in time input
        if len(st) == 3 and len(end) == 3:
            df1 = df[(df.index >= pd.datetime(st[0], st[1], st[2])) & (df.index <= pd.datetime(end[0], end[1], end[2]))]
        else:
            df1 = df[(df.index >= pd.datetime(st[0], st[1], st[2], st[3], st[4])) & (
            df.index <= pd.datetime(end[0], end[1], end[2], st[3], st[4]))]

        # account for hours in time input
        if excs[0] == 0:
            pass
        else:
            if len(excs) == 3:
                df1 = df1[(df1.index < pd.datetime(excs[0], excs[1], excs[2])) | (
                df1.index > pd.datetime(excf[0], excf[1], excf[2]))]
            else:
                df1 = df1[(df1.index < pd.datetime(excs[0], excs[1], excs[2], excs[3], excs[4])) | (
                df1.index > pd.datetime(excf[0], excf[1], excf[2], excf[3], excf[4]))]

        df2 = df1.dropna(subset=[Q])

        y1 = df2[Q]
        x1 = (df2.index.to_julian_date() - df2.index.to_julian_date()[0])  # convert to numeric days for opt. function
        popt1, pcov1 = self.fitit(x1, y1, y1[0])  # fit curve
        x2 = [df2.index[0] + timedelta(i) for i in x1]  # convert back to dates for labels
        y2 = [y1[0] * np.exp(-1 * popt1[0] * i) for i in x1]  # run function with optimized variables
        plt.plot(x2, y2, label='Recession (alpha = %.3f)' % popt1[0])  # report alpha value
        plt.scatter(x2, y1, label='Discharge')
        plt.ylabel(self.Qlab)
        plt.legend(scatterpoints=1)
        plt.show()
        return popt1, x1, x2, y1, y2


class piper(object):
    """Class that generates rectangular piper diagrams.

    :param df: DataFrame containing chemistry data; must have fields labeled as abbreviations of the major ions; Na, K,
    NaK, Ca, Mg, Cl, HCO3, CO3, and SO4
    :type df: pandas.core.frame.DataFrame
    :param type_col: Name of field that designates the sample type (optional); defaults to ''
    :type type_col: str
    :param var_col: Name of field that contains a scalar variable to be designated by color (optional); defaults to ''
    :type var_col: str


    .. note::
    Hydrochemistry - Construct Rectangular Piper plot
    Adopted from: Ray and Mukherjee, 2008, Groundwater 46(6): 893-896 and from code found at:
    http://python.hydrology-amsterdam.nl/scripts/piper_rectangular.py
    Based on code by:
    B.M. van Breukelen <b.m.vanbreukelen@vu.nl>  
      
    """

    def __init__(self, df, type_col='', var_col=''):

        self.fieldnames = [u'Na', u'K', u'NaK', u'Ca', u'Mg', u'Cl', u'HCO3', u'CO3', u'SO4']
        self.anions = ['Cl', 'HCO3', 'CO3', 'SO4']
        self.cations = ['Na', 'K', 'Ca', 'Mg', 'NaK']
        self.piperplot(df,  type_col, var_col)


    def fillMissing(self, df):

        # fill in nulls with 0
        for col in df.columns:
            if col in self.fieldnames:
                for i in range(len(df)):
                    if df.loc[i, col] is None or df.loc[i, col] == '' or np.isnan(df.loc[i, col]):
                        df.loc[i, col] = 0
            else:
                df.col = 0

        # add missing columns
        for name in self.fieldnames:
            if name in df.columns:
                pass
            else:
                print(name)
                df[name] = 0

        return df

    def check_nak(self, x):
        if x[0] == 0 and x[2] > 0:
            return x[2]
        else:
            return x[0] + x[1]

    def convertIons(self, df):
        """Convert major ion concentrations from mg/L to meq

        This function uses conversion factors to convert the concentrations of major ions from mg/L to meq.  It also
        appends a field to the input database listing the Cation-Anion pair that have the highest meq concentrations.

        :param df: DataFrame containing chemistry data; must have fields labeled as abbreviations of the major ions; Na, K,
        NaK, Ca, Mg, Cl, HCO3, CO3, and SO4
        :returns: appends convert fields onto DataFrame with the suffix `_meq` and adds the fields 'water type', 'CBE'
        (charge balance), 'EC' (Sum(anions+cations))

        """
        # Conversion factors from mg/L to meq/L
        d = {'Ca': 0.04990269, 'Mg': 0.082287595, 'Na': 0.043497608, 'K': 0.02557656, 'Cl': 0.028206596, 'NaK': 0.043497608,
             'HCO3': 0.016388838, 'CO3': 0.033328223, 'SO4': 0.020833333, 'NO2': 0.021736513, 'NO3': 0.016129032}

        df1 = df

        for name in self.fieldnames:
            if name in df.columns:
                df1[name + '_meq'] = df1[name].apply(lambda x: float(d.get(name, 0)) * x, 1)


        df1['NaK_meq'] = df1[['Na_meq', 'K_meq','NaK_meq']].apply(lambda x: self.check_nak(x), 1)

        df1['anions'] = 0
        df1['cations'] = 0

        for ion in self.anions:
            if ion in df.columns:
                df1['anions'] += df1[ion + '_meq']
        for ion in self.cations:
            if ion in df1.columns:
                df1['cations'] += df1[ion + '_meq']

        df1['EC'] = df1['anions'] - df1['cations']
        df1['CBE'] = df1['EC'] / (df1['anions'] + df1['cations'])
        df1['maj_cation'] = df1[['Ca_meq','Mg_meq','Na_meq','K_meq','NaK_meq']].idxmax(axis=1)
        df1['maj_anion'] = df1[['Cl_meq','SO4_meq','HCO3_meq','CO3_meq']].idxmax(axis=1)
        df1['water_type'] = df1[['maj_cation','maj_anion']].apply(lambda x: str(x[0])[:-4]+'-'+str(x[1])[:-4],1)
        return df1

    def ionPercentage(self, df):
        """Determines percentage of charge for each ion for display on the piper plot"""
        for ion in self.anions:
            df[ion + 'EC'] = df[[ion + '_meq', 'anions']].apply(lambda x: 100 * x[0] / x[1], 1)
        for ion in self.cations:
            df[ion + 'EC'] = df[[ion + '_meq', 'cations']].apply(lambda x: 100 * x[0] / x[1], 1)
        return df

    def piperplot(self, df,  type_col, var_col):
        """Generates a rectangular piper diagram"""
        self.fillMissing(df)
        self.convertIons(df)
        self.ionPercentage(df)

        CaEC = df['CaEC'].values
        MgEC = df['MgEC'].values
        ClEC = df['ClEC'].values
        SO4EC = df['SO4EC'].values
        NaKEC = df['NaKEC'].values
        SO4ClEC = df[['ClEC', 'SO4EC']].apply(lambda x: x[0] + x[1], 1).values
        
        num_samps = len(df)
        if var_col == '':
            Elev = ''
        else:
            Elev = df[var_col].values

        if type_col == '':
            typ = ['Station']*num_samps
            stationtypes = ['Station']
        else:
            stationtypes = list(df[type_col].unique())
            typ = df[type_col].values
                            
        # Change default settings for figures
        plt.rc('xtick', labelsize=10)
        plt.rc('ytick', labelsize=10)
        plt.rc('font', size=12)
        plt.rc('legend', fontsize=12)
        plt.rc('figure', figsize=(14, 5.5))  # defines size of Figure window orig (14,4.5)

        markSize = 30
        lineW = 0.5

        # Make Figure
        fig = plt.figure()
        # add title
        # fig.suptitle(piperTitle, x=0.20,y=.98, fontsize=14 )
        # Colormap and Saving Options for Figure

        if len(Elev) > 0:
            vart = Elev
        else:
            vart = [1] * num_samps
        cNorm = plt.Normalize(vmin=min(vart), vmax=max(vart))
        cmap = plt.cm.coolwarm
        # pdf = PdfPages(fileplace)

        mrkrSymbl = ['v', '^', '+', 's', '.', 'o', '*', 'v', '^', '+', 's', ',', '.', 'o', '*', 'v', '^', '+', 's', ',',
                     '.', 'o', '*', 'v', '^', '+', 's', ',', '.', 'o', '*']

        # count variable for legend (n)
        unique, counts = np.unique(typ, return_counts=True)
        nstatTypesDict = dict(zip(unique, counts))
 
        typdict = {}
        for i in range(len(stationtypes)):
            typdict[stationtypes[i]] = mrkrSymbl[i]
 
        # CATIONS-----------------------------------------------------------------------------
        # 2 lines below needed to create 2nd y-axis (ax1b) for first subplot
        ax1 = fig.add_subplot(131)
        ax1b = ax1.twinx()

        ax1.fill([100, 0, 100, 100], [0, 100, 100, 0], color=(0.8, 0.8, 0.8))
        ax1.plot([100, 0], [0, 100], 'k')
        ax1.plot([50, 0, 50, 50], [0, 50, 50, 0], 'k--')
        ax1.text(25, 15, 'Na type')
        ax1.text(75, 15, 'Ca type')
        ax1.text(25, 65, 'Mg type')

        if len(typ) > 0:
            for j in range(len(typ)):
                ax1.scatter(CaEC[j], MgEC[j], s=markSize, c=vart[j], cmap=cmap, norm=cNorm, marker=typdict[typ[j]],
                            linewidths=lineW)
        else:
            ax1.scatter(CaEC, MgEC, s=markSize, c=vart, cmap=cmap, norm=cNorm, linewidths=lineW)

        ax1.set_xlim(0, 100)
        ax1.set_ylim(0, 100)
        ax1b.set_ylim(0, 100)
        ax1.set_xlabel('<= Ca (% meq)')
        ax1b.set_ylabel('Mg (% meq) =>')
        plt.setp(ax1, yticklabels=[])

        # next line needed to reverse x axis:
        ax1.set_xlim(ax1.get_xlim()[::-1])

        # ANIONS----------------------------------------------------------------------------
        ax = fig.add_subplot(1, 3, 3)
        ax.fill([100, 100, 0, 100], [0, 100, 100, 0], color=(0.8, 0.8, 0.8))
        ax.plot([0, 100], [100, 0], 'k')
        ax.plot([50, 50, 0, 50], [0, 50, 50, 0], 'k--')
        ax.text(55, 15, 'Cl type')
        ax.text(5, 15, 'HCO3 type')
        ax.text(5, 65, 'SO4 type')

        if len(typ) > 0:
            for j in range(len(typ)):
                labs = "{:} n= {:}".format(typ[j],nstatTypesDict[typ[j]])
                if float(nstatTypesDict[typ[j]]) > 1:
                    s = ax.scatter(ClEC[j], SO4EC[j], s=markSize, c=vart[j], cmap=cmap, norm=cNorm,
                                   marker=typdict[typ[j]], label=labs, linewidths=lineW)
                else:
                    s = ax.scatter(ClEC[j], SO4EC[j], s=markSize, c=vart[j], cmap=cmap, norm=cNorm,
                                   marker=typdict[typ[j]], label=typ[j], linewidths=lineW)
        else:
            s = ax.scatter(ClEC, SO4EC, s=markSize, c=vart, cmap=cmap, norm=cNorm, label='Sample', linewidths=lineW)

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel('Cl (% meq) =>')
        ax.set_ylabel('SO4 (% meq) =>')

        # CATIONS AND ANIONS COMBINED ---------------------------------------------------------------
        # 2 lines below needed to create 2nd y-axis (ax1b) for first subplot
        ax2 = fig.add_subplot(132)
        ax2b = ax2.twinx()

        ax2.plot([0, 100], [10, 10], 'k--')
        ax2.plot([0, 100], [50, 50], 'k--')
        ax2.plot([0, 100], [90, 90], 'k--')
        ax2.plot([10, 10], [0, 100], 'k--')
        ax2.plot([50, 50], [0, 100], 'k--')
        ax2.plot([90, 90], [0, 100], 'k--')

        if len(typ) > 0:
            for j in range(len(typ)):
                ax2.scatter(NaKEC[j], SO4ClEC[j], s=markSize, c=vart[j], cmap=cmap, norm=cNorm, marker=typdict[typ[j]],
                            linewidths=lineW)
        else:
            ax2.scatter(NaKEC, SO4ClEC, s=markSize, c=vart, cmap=cmap, norm=cNorm, linewidths=lineW)

        ax2.set_xlim(0, 100)
        ax2.set_ylim(0, 100)
        ax2.set_xlabel('Na+K (% meq) =>')
        ax2.set_ylabel('SO4+Cl (% meq) =>')
        ax2.set_title('<= Ca+Mg (% meq)', fontsize=12)
        ax2b.set_ylabel('<= CO3+HCO3 (% meq)')
        ax2b.set_ylim(0, 100)

        # next two lines needed to reverse 2nd y axis:
        ax2b.set_ylim(ax2b.get_ylim()[::-1])

        # Align plots
        plt.subplots_adjust(left=0.05, bottom=0.35, right=0.95, top=0.90, wspace=0.4, hspace=0.0)

        # Legend-----------------------------------------------------------------------------------------

        # Add colorbar below legend
        # [left, bottom, width, height] where all quantities are in fractions of figure width and height

        if len(typ) > 0:
            handles, labels = ax.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))

            plt.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=5, shadow=False, fancybox=True,
                       bbox_to_anchor=(0.5, -0.3), scatterpoints=1)

        if len(Elev) > 0:
            cax = fig.add_axes([0.25, 0.10, 0.50, 0.02])
            cb1 = plt.colorbar(s, cax=cax, cmap=cmap, norm=cNorm, orientation='horizontal')  # use_gridspec=True
            cb1.set_label(var_col, size=8)

        self.plot = fig
        self.df = df

        
def fdc(df, site, begyear=1900, endyear=2015, normalizer=1, plot=True):
    """Generate flow duration curve for hydrologic time series data

    :param df: DataFrame with discharge data of interest; must have a date or date-time as the index
    :type df: pandas.core.frame.DataFrame
    :param site: Name of DataFrame column in df containing discharge data
    :type site: str
    :param begyear: beginning year of analysis; defaults to 1900
    :type begyear: int
    :param endyear: end year of analysis; defaults to 2015
    :type endyear: int
    :param normalizer: value to use to normalize discharge; defaults to 1 (no normalization)
    :type normalizer: int
    :param plot: Whether to generate the plot or just return the variables for a plot; defaults to true
    :type plot: bool

    :returns: matplotlib plot displaying the flow duration curve of the data
    :return prob: x field stating the probability of a discharge in data
    :rtype prob: list
    :return data: y field stating the discharge for probability prob
    :rtype data: list

    """
    # limit dataframe to only the site
    df = df[[site]]

    # filter dataframe to only include dates of interest
    data = df[
        (df.index.to_datetime() > pd.datetime(begyear, 1, 1)) & (df.index.to_datetime() < pd.datetime(endyear, 1, 1))]

    # remove na values from dataframe
    data = data.dropna()

    # take average of each day of year (from 1 to 366) over the selected period of record
    data['doy'] = data.index.dayofyear
    dailyavg = data[site].groupby(data['doy']).mean()

    data = np.sort(dailyavg)

    ## uncomment the following to use normalized discharge instead of discharge
    # mean = np.mean(data)
    # std = np.std(data)
    # data = [(data[i]-np.mean(data))/np.std(data) for i in range(len(data))]
    data = [(data[i]) / normalizer for i in range(len(data))]

    # ranks data from smallest to largest
    ranks = sp.rankdata(data, method='average')

    # reverses rank order
    ranks = ranks[::-1]

    # calculate probability of each rank
    prob = [(ranks[i] / (len(data) + 1)) for i in range(len(data))]

    # plot data via matplotlib
    if plot:
        plt.plot(prob, data, label=site + ' ' + str(begyear) + '-' + str(endyear))
    else:
        pass
    return prob, data


class gantt(object):
    """Class to create gantt plots and to summarize pandas timeseries dataframes.

    Finds gaps and measuring duration of data.
    :param df: The DataFrame with a datetime index and columns as site time-series data; each column name
    should be the site name or the site labels should be input for chart
    :param stations: List of columns to include in the chart; defaults to all columns
    :param labels: Labels to use in the resulting plot for each station; must be equal to the length of stations list;
    defaults to stations
    :param samp_int: regular interval that the datetime index will be resampled. Defaults to daily ('D');
    see http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases for all of the options
    :type samp_int: str
    :type df: pandas.core.frame.DataFrame
    :type stations: list
    :type labels: list

    .. note::
    `.stations` produces a list describing the stations put into the class
    `.labels` produces a list describing the labels put into the class
    `.dateranges` is a dictionary describing gaps in the dataframe based on the presence of nan values in the frame
    `.ganttPlotter()` plots a gantt plot

    """

    def __init__(self, df, stations=[], labels=[], samp_int = 'D'):
        if len(stations) == 0:
            stations = df.columns
        if len(labels) == 0:
            labels = stations

        self.data = df.resample(samp_int).mean()
        self.stations = stations
        self.labels = labels
        self.dateranges = self.markGaps()
        self.sitestats = self.site_info()
        print(
        'Data Loaded \nType .ganttPlotter() after your defined object to make plot\nType .sitestats after your defined object to get summary stats')

    def markGaps(self):
        """Produces dictionary of list of gaps in time series data based on the presence of nan values;
        used for gantt plotting

        :returns: dateranges; a dictionary with station names as keys and lists of begin and end dates as values
        """
        df = self.data
        stations = self.stations

        dateranges = {}
        for station in stations:
            dateranges[station] = []
            first = df.ix[:, station].first_valid_index()
            last = df.ix[:, station].last_valid_index()
            records = df.ix[first:last, station]
            #dateranges[station].append(pd.to_datetime(first))
            for i in range(len(records) - 1):
                if pd.isnull(records[i + 1]) and pd.notnull(records[i]):
                    dateranges[station].append(pd.to_datetime(records.index)[i])
                elif pd.isnull(records[i]) and pd.notnull(records[i + 1]):
                    dateranges[station].append(pd.to_datetime(records.index)[i])
            dateranges[station].append(pd.to_datetime(last))
        return dateranges

    def site_info(self):
        """Creates a table of summary statistics for all of the stations in the stations field defined in the class

        :returns: site_info; a table of summary statistics; first, last, min, max, std, median, avg, 25%tile, 75%tile,
        and count

        """
        stations = self.stations
        df = self.data

        stat, first, last, minum, maxum, stdev, medin, avg, q25, q75, count = [], [], [], [], [], [], [], [], [], [], []
        for station in stations:
            stdt = df.ix[:, station]
            stat.append(station)
            first.append(stdt.first_valid_index())
            last.append(stdt.last_valid_index())
            minum.append(stdt.min())
            maxum.append(stdt.max())
            stdev.append(stdt.std())
            medin.append(stdt.median())
            avg.append(stdt.mean())
            q25.append(stdt.quantile(0.25))
            q75.append(stdt.quantile(0.75))
            count.append(stdt.count())
        colm = {'StationId': stat, 'first': first, 'last': last, 'min': minum, 'max': maxum,
                'std': stdev, 'median': medin, 'mean': avg, 'q25': q25, 'q75': q75, 'count': count}
        Site_Info = pd.DataFrame(colm)
        return Site_Info

    def ganttPlotter(self):
        """Plots gantt plot using dictionary of stations and associated start and end dates;
        uses output from markGaps function"""

        labs, tickloc, col = [], [], []

        dateranges = self.dateranges
        stations = self.stations
        labels = self.labels

        # create color iterator for multi-color lines in gantt chart
        color = iter(plt.cm.Dark2(np.linspace(0, 1, len(stations))))

        plt.figure(figsize=[8, 10])
        fig, ax = plt.subplots()

        for i in range(len(stations)):
            c = next(color)
            for j in range(len(dateranges[stations[i]]) - 1):
                if (j + 1) % 2 != 0:
                    if len(labels) == 0 or len(labels) != len(stations):
                        plt.hlines(i + 1, dateranges[stations[i]][j], dateranges[stations[i]][j + 1], label=stations[i],
                                   color=c, linewidth=3)
                    else:
                        plt.hlines(i + 1, dateranges[stations[i]][j], dateranges[stations[i]][j + 1], label=labels[i],
                                   color=c, linewidth=3)
            labs.append(stations[i])
            tickloc.append(i + 1)
            col.append(c)
        plt.ylim(0, len(stations) + 1)

        if len(labels) == 0 or len(labels) != len(stations):
            labels = stations
            plt.yticks(tickloc, labs)
        else:
            plt.yticks(tickloc, labels)

        plt.xlabel('Date')
        plt.ylabel('Station Name')
        plt.grid(linewidth=0.2)

        gytl = plt.gca().get_yticklabels()
        for i in range(len(gytl)):
            gytl[i].set_color(col[i])
        plt.tight_layout()
        return fig

    def gantt(self):
        """This function runs the other functions in this class."""
        stations = self.stations
        labels = self.labels
        df = self.data

        df1 = df.ix[:, stations]
        df1.sort_index(inplace=True)
        Site_Info = self.site_info()
        dateranges = self.markGaps()
        fig = self.ganttPlotter()
        return Site_Info, dateranges, fig


def scatterColor(x0, y, w):
    """Creates scatter plot with points colored by variable.
    All input arrays must have matching lengths

    :param x0: x values to plot
    :type x0: list
    :param y: y values to plot
    :type y: list
    :param w: z values to plot

    :returns: plot; slope and intercept of the RLM best fit line shown on the plot
    .. warning:: all input arrays must have matching lengths and scalar values
    .. note:: See documentation at http://statsmodels.sourceforge.net/0.6.0/generated/statsmodels.robust.robust_linear_model.RLM.html
    for the RLM line
    """
    import matplotlib as mpl
    import matplotlib.cm as cm
    import statsmodels.api as sm
    from scipy.stats import linregress
    cmap = plt.cm.get_cmap('RdYlBu')
    norm = mpl.colors.Normalize(vmin=w.min(), vmax=w.max())
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    m.set_array(w)
    sc = plt.scatter(x0, y, label='', color=m.to_rgba(w))

    xa = sm.add_constant(x0)

    est = sm.RLM(y, xa).fit()
    r2 = sm.WLS(y, xa, weights=est.weights).fit().rsquared
    slope = est.params[1]

    x_prime = np.linspace(np.min(x0), np.max(x0), 100)[:, np.newaxis]
    x_prime = sm.add_constant(x_prime)
    y_hat = est.predict(x_prime)

    const = est.params[0]
    y2 = [i * slope + const for i in x0]

    lin = linregress(x0, y)
    x1 = np.arange(np.min(x0), np.max(x0), 0.1)
    y1 = [i * lin[0] + lin[1] for i in x1]
    y2 = [i * slope + const for i in x1]
    plt.plot(x1, y1, c='g',
             label='simple linear regression m = {:.2f} b = {:.0f}, r^2 = {:.2f}'.format(lin[0], lin[1], lin[2] ** 2))
    plt.plot(x1, y2, c='r', label='rlm regression m = {:.2f} b = {:.0f}, r2 = {:.2f}'.format(slope, const, r2))
    plt.legend()
    cbar = plt.colorbar(m)

    cbar.set_label('use cbar.set_label("label") to label this axis')

    return slope, const

