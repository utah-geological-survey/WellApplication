# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 12:32:51 2015

@author: paulinkenbrandt
"""
import scipy
from scipy import stats as sp
#import scipy.stats as sp
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
from datetime import datetime, timedelta

class piper:
    """
    Created on Thu May 29 10:57:49 2014

    Hydrochemistry - Construct Rectangular Piper plot

    Adopted from: Ray and Mukherjee (2008) Groundwater 46(6): 893-896 
    and from code found at:
    http://python.hydrology-amsterdam.nl/scripts/piper_rectangular.py

    Based on code by:
    B.M. van Breukelen <b.m.vanbreukelen@vu.nl>  
      
    """
    def __init__(self):
        
        self.fieldnames = [u'Na', u'K', u'Ca', u'Mg', u'Cl', u'HCO3', u'CO3', u'SO4']
        self.anions = ['Cl','HCO3','CO3','SO4']
        self.cations = ['Na','K','Ca','Mg','NaK']
        print('ok')
    
    def fillMissing(self, df):
        
        # fill in nulls with 0
        for col in df.columns:
            if col in self.fieldnames:
                for i in range(len(df)):
                    if df.loc[i,col] == None or df.loc[i,col]=='' or np.isnan(df.loc[i,col]):
                        df.loc[i,col] = 0
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
    
    def convertIons(self, df):
        
        #Conversion factors from mg/L to meq/L
        d = {'Ca':0.04990269, 'Mg':0.082287595, 'Na':0.043497608, 'K':0.02557656, 'Cl':0.028206596, 'HCO3':0.016388838, 'CO3':0.033328223, 'SO4':0.020833333, 'NO2':0.021736513, 'NO3':0.016129032}
    
        df1 = df
        
        for name in self.fieldnames:
            if name in df.columns:
                df1[name + '_meq'] = df1[name].apply(lambda x: float(d.get(name,0))*x ,1)
       
        if 'Na_meq' in df1.columns and 'K_meq' in df1.columns:
            df1['NaK_meq'] = df1[['Na_meq','K_meq']].apply(lambda x: x[0]+x[1],1)
        
        df1['anions'] = 0
        df1['cations'] = 0
        
        for ion in self.anions:
            if ion in df.columns:
                df1['anions'] += df1[ion+'_meq']
        for ion in self.cations:
            if ion in df1.columns:
                df1['cations'] += df1[ion+'_meq']
        
        df1['EC'] = df1['anions'] - df1['cations']
        df1['CBE'] = df1['EC']/(df1['anions'] + df1['cations'])
        
        return df1

            
    def ionPercentage(self, df):
    
        for ion in self.anions:
            df[ion+'EC'] = df[[ion+'_meq','anions']].apply(lambda x: 100*x[0]/x[1],1)
        for ion in self.cations:
            df[ion+'EC'] = df[[ion+'_meq','cations']].apply(lambda x: 100*x[0]/x[1],1)
        
        return df
    
    
    def piperplot(self, df):
        
        self.fillMissing(df)
        self.convertIons(df)
        self.ionPercentage(df)
        
        CaEC = df['CaEC'].values
        MgEC = df['MgEC'].values
        ClEC = df['ClEC'].values
        SO4EC = df['SO4EC'].values
        NaKEC = df['NaKEC'].values
        SO4ClEC = df[['ClEC','SO4EC']].apply(lambda x: x[0]+x[1],1).values
        
        Elev = len(df)*[0] # Fix this
        stationtypes= list(df['type'].unique())
        
        # Change default settings for figures
        plt.rc('xtick', labelsize = 10)
        plt.rc('ytick', labelsize = 10)
        plt.rc('font', size = 12)
        plt.rc('legend', fontsize = 12)
        plt.rc('figure', figsize = (14,5.5)) # defines size of Figure window orig (14,4.5)
        
        markSize = 30
        lineW = 0.5

        # Make Figure
        fig = plt.figure()
        # add title
        #fig.suptitle(piperTitle, x=0.20,y=.98, fontsize=14 )
        # Colormap and Saving Options for Figure
        
        if len(Elev)>0:
            vart = Elev
        else:
            vart = [1]*len(df)
        cNorm  = plt.Normalize(vmin=min(vart), vmax=max(vart))
        cmap = plt.cm.coolwarm
        #pdf = PdfPages(fileplace)
        
        mrkrSymbl = ['v', '^', '+', 's', '.', 'o', '*', 'v', '^', '+', 's', ',', '.', 'o', '*','v', '^', '+', 's', ',', '.', 'o', '*', 'v', '^', '+', 's', ',', '.', 'o', '*']
        
        # count variable for legend (n)
        #nstatTypes = len(list(set(stationtypes)))
        typeSet = [0]*len(stationtypes)
        #nstatTypes = [typ.count(i) for i in stationtypes]
        typ =[]
        typdict = {}
        nstatTypesDict = {}
        for i in range(len(stationtypes)):
            typdict[stationtypes[i]] = mrkrSymbl[i]
            nstatTypesDict[stationtypes[i]] = str(typeSet[i])
        
        # CATIONS-----------------------------------------------------------------------------
        # 2 lines below needed to create 2nd y-axis (ax1b) for first subplot
        ax1 = fig.add_subplot(131)
        ax1b = ax1.twinx()
        
        ax1.fill([100,0,100,100],[0,100,100,0],color = (0.8,0.8,0.8))
        ax1.plot([100, 0],[0, 100],'k')
        ax1.plot([50, 0, 50, 50],[0, 50, 50, 0],'k--')
        ax1.text(25,15, 'Na type')
        ax1.text(75,15, 'Ca type')
        ax1.text(25,65, 'Mg type')
        
        if len(typ) > 0:
            for j in range(len(typ)):    
                ax1.scatter(CaEC[j], MgEC[j], s=markSize, c=vart[j], cmap= cmap, norm = cNorm, marker=typdict[typ[j]], linewidths = lineW)
        else:
            ax1.scatter(CaEC, MgEC, s=markSize, c=vart, cmap= cmap, norm = cNorm, linewidths = lineW)
        
        ax1.set_xlim(0,100)
        ax1.set_ylim(0,100)
        ax1b.set_ylim(0,100)
        ax1.set_xlabel('<= Ca (% meq)')
        ax1b.set_ylabel('Mg (% meq) =>')
        plt.setp(ax1, yticklabels=[])
        
        # next line needed to reverse x axis:
        ax1.set_xlim(ax1.get_xlim()[::-1]) 
        
        # ANIONS----------------------------------------------------------------------------
        ax = fig.add_subplot(1,3,3)
        ax.fill([100,100,0,100],[0,100,100,0],color = (0.8,0.8,0.8))
        ax.plot([0, 100],[100, 0],'k')
        ax.plot([50, 50, 0, 50],[0, 50, 50, 0],'k--')
        ax.text(55,15, 'Cl type')
        ax.text(5,15, 'HCO3 type')
        ax.text(5,65, 'SO4 type')
        
        if len(typ) > 0:
            for j in range(len(typ)):
                labs = typ[j] + " n= " + nstatTypesDict[typ[j]]
                if float(nstatTypesDict[typ[j]]) > 1:
                    s = ax.scatter(ClEC[j], SO4EC[j], s=markSize, c=vart[j], cmap=cmap, norm =cNorm, marker=typdict[typ[j]], label=labs, linewidths = lineW)
                else:
                    s = ax.scatter(ClEC[j], SO4EC[j], s=markSize, c=vart[j], cmap=cmap, norm =cNorm, marker=typdict[typ[j]], label=typ[j], linewidths = lineW)
        else:
            s = ax.scatter(ClEC, SO4EC, s=markSize, c=vart, cmap=cmap, norm =cNorm, label='Sample', linewidths = lineW)
        
        ax.set_xlim(0,100)
        ax.set_ylim(0,100)
        ax.set_xlabel('Cl (% meq) =>')
        ax.set_ylabel('SO4 (% meq) =>')
        
        # CATIONS AND ANIONS COMBINED ---------------------------------------------------------------
        # 2 lines below needed to create 2nd y-axis (ax1b) for first subplot
        ax2 = fig.add_subplot(132)
        ax2b = ax2.twinx()
        
        ax2.plot([0, 100],[10, 10],'k--')
        ax2.plot([0, 100],[50, 50],'k--')
        ax2.plot([0, 100],[90, 90],'k--')
        ax2.plot([10, 10],[0, 100],'k--')
        ax2.plot([50, 50],[0, 100],'k--')
        ax2.plot([90, 90],[0, 100],'k--')
        
        if len(typ) > 0:
            for j in range(len(typ)):    
                ax2.scatter(NaKEC[j], SO4ClEC[j], s=markSize, c=vart[j], cmap=cmap, norm =cNorm, marker=typdict[typ[j]], linewidths = lineW)
        else:
            ax2.scatter(NaKEC, SO4ClEC, s=markSize, c=vart, cmap=cmap, norm =cNorm, linewidths = lineW)
        
        ax2.set_xlim(0,100)
        ax2.set_ylim(0,100)
        ax2.set_xlabel('Na+K (% meq) =>')
        ax2.set_ylabel('SO4+Cl (% meq) =>')
        ax2.set_title('<= Ca+Mg (% meq)', fontsize = 12)
        ax2b.set_ylabel('<= CO3+HCO3 (% meq)')
        ax2b.set_ylim(0,100)
        
        # next two lines needed to reverse 2nd y axis:
        ax2b.set_ylim(ax2b.get_ylim()[::-1])
        
        # Align plots
        plt.subplots_adjust(left=0.05, bottom=0.35, right=0.95, top=0.90, wspace=0.4, hspace=0.0)    
        
        #Legend-----------------------------------------------------------------------------------------
        
        # Add colorbar below legend
        #[left, bottom, width, height] where all quantities are in fractions of figure width and height
        
        
        if len(Elev)>0:
            cax = fig.add_axes([0.25,0.10,0.50,0.02])    
            cb1 = plt.colorbar(s, cax=cax, cmap=cmap, norm=cNorm, orientation='horizontal') #use_gridspec=True
            cb1.set_label("Test",size=8) 
            
        if len(typ)>0:
            handles, labels = ax.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
        
            plt.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=5, shadow=False, fancybox=True, bbox_to_anchor=(0.5, 0.6), scatterpoints=1)
        
        self.plot = fig
        self.df = df
        
class fdc:
    
    @staticmethod
    def fdc(df,site,begyear,endyear):
        '''
        Generate flow duration curve for hydrologic time series data
        
        df = pandas dataframe containing data
        site = column within dataframe that contains the flow values
        begyear = start year for analysis
        endyear = end year for analysis
        '''
            
        data = df[(df.index.to_datetime() > pd.datetime(begyear,1,1))&(df.index.to_datetime() < pd.datetime(endyear,1,1))]
        data = data[site].dropna().values
        data = np.sort(data)
        ranks = sp.rankdata(data, method='average')
        ranks = ranks[::-1]
        prob = [100*(ranks[i]/(len(data)+1)) for i in range(len(data)) ]
        plt.figure()
        plt.scatter(prob,data,label=site)
        plt.yscale('log')
        plt.grid(which = 'both')
        plt.xlabel('% of time that indicated discharge was exceeded or equaled')
        plt.ylabel('discharge (cfs)')
        plt.xticks(range(0,100,5))
        plt.title('Flow duration curve for ' + site)
    
    @staticmethod
    def fdc_simple(df, site, begyear=1900, endyear=2015, normalizer=1):
        '''
        Generate flow duration curve for hydrologic time series data
        
        PARAMETERS:
            df = pandas dataframe of interest; must have a date or date-time as the index
            site = pandas column containing discharge data; must be within df
            begyear = beginning year of analysis; defaults to 1900
            endyear = end year of analysis; defaults to 2015
            normalizer = value to use to normalize discharge; defaults to 1 (no normalization)
        
        RETURNS:
            matplotlib plot displaying the flow duration curve of the data
            
        REQUIRES:
            numpy as np
            pandas as pd
            matplotlib.pyplot as plt
            scipy.stats as sp
        '''
        # limit dataframe to only the site
        df = df[[site]]
        
        # filter dataframe to only include dates of interest
        data = df[(df.index.to_datetime() > pd.datetime(begyear,1,1))&(df.index.to_datetime() < pd.datetime(endyear,1,1))]
    
        # remove na values from dataframe
        data = data.dropna()
    
        # take average of each day of year (from 1 to 366) over the selected period of record
        data['doy']=data.index.dayofyear
        dailyavg = data[site].groupby(data['doy']).mean()
            
        data = np.sort(dailyavg)
    
        ## uncomment the following to use normalized discharge instead of discharge
        #mean = np.mean(data)
        #std = np.std(data)
        #data = [(data[i]-np.mean(data))/np.std(data) for i in range(len(data))]
        data = [(data[i])/normalizer for i in range(len(data))]
        
        # ranks data from smallest to largest
        ranks = sp.rankdata(data, method='average')
    
        # reverses rank order
        ranks = ranks[::-1]
        
        # calculate probability of each rank
        prob = [(ranks[i]/(len(data)+1)) for i in range(len(data)) ]
        
        # plot data via matplotlib
        plt.plot(prob,data,label=site+' '+str(begyear)+'-'+str(endyear))
        
def gantt(df, stations = [], labels = []):
    '''
    INPUT
    -----
    df = pandas dataframe with datetime as index and columns as site time-series data; each column name should be the site name
    sites = list of columns you want to subset from your dataframe
    
    RETURNS
    -------
    gantt chart and site info table
    
    '''
    if len(stations) == 0:
        stations = df.columns
    
    q = {}
    m = {}
    for site in stations:
        if site in df.columns:
            q[site] = df[site].first_valid_index()
            m[site] = df[site].last_valid_index()
    
    start_date = pd.DataFrame(data=q, index=[0])
    finish_date = pd.DataFrame(data=m, index=[0])
    start_date = start_date.transpose()
    start_date['start_date'] = start_date[0]
    start_date = start_date.drop([0],axis=1)
    finish_date = finish_date.transpose()
    finish_date['fin_date'] = finish_date[0]
    finish_date = finish_date.drop([0],axis=1)
    start_fin = pd.merge(finish_date, start_date, left_index=True, right_index=True, how='inner' )

    sum_stats = df[stations].describe()
    sum_stats = sum_stats.transpose()
    Site_Info = pd.merge(sum_stats, start_fin, left_index=True, right_index=True, how='inner' )
    
    dateranges = {}
    for station in stations:
        dateranges[station] = []
        first = df.ix[:,station].first_valid_index()
        last =  df.ix[:,station].last_valid_index()
        records = df.ix[first:last,station]
        dateranges[station].append(pd.to_datetime(first))
        for i in range(len(records)-1):
            if np.isnan(records[i+1]) and np.isfinite(records[i]):
                dateranges[station].append(pd.to_datetime(records.index)[i])
            elif np.isnan(records[i]) and np.isfinite(records[i+1]):
                dateranges[station].append(pd.to_datetime(records.index)[i])
        dateranges[station].append(pd.to_datetime(last))

    labs, tickloc, col = [], [], []

    # create color iterator for multi-color lines in gantt chart
    color = iter(plt.cm.Dark2(np.linspace(0,1,len(stations))))

    plt.figure(figsize=[8,10])
    fig, ax = plt.subplots()

    for i in range(len(stations)):
        c=next(color)
        for j in range(len(dateranges[stations[i]])-1):
            if (j+1)%2 != 0:
                if len(labels) == 0 or len(labels)!=len(stations):
                    plt.hlines(i+1, dateranges[stations[i]][j], dateranges[stations[i]][j+1], label = stations[i], color=c, linewidth=3)
                else:
                    plt.hlines(i+1, dateranges[stations[i]][j], dateranges[stations[i]][j+1], label = labels[i], color=c, linewidth=3)
        labs.append(stations[i])
        tickloc.append(i+1)
        col.append(c)
    plt.ylim(0,len(stations)+1)

    if len(labels) == 0 or len(labels)!=len(stations):
        labels = stations
        plt.yticks(tickloc, labs)
    else:
        plt.yticks(tickloc, labels)
    
    plt.xlabel('Date')
    plt.ylabel('Station Name')
    plt.grid(linewidth=0.2)
    #plt.title('USGS Station Measurement Duration')
    # color y labels to match lines
    gytl = plt.gca().get_yticklabels()
    for i in range(len(gytl)):
        gytl[i].set_color(col[i])
    plt.tight_layout()
        
    return Site_Info, fig
