# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 00:30:36 2016

@author: p
"""
import urllib2
import xmltodict
import pandas as pd
from datetime import datetime 
from httplib import BadStatusLine
import matplotlib.pyplot as plt
import numpy as np
import avgMeths

class usgs:
    @staticmethod 
    def getelev(x):
        '''
        Uses USGS elevation service to retrieve elevation
        
        Input
        -----
        x, y = longitude and latitude of point where elevation is desired        
    
        Output
        ------
        ned float elevation of location in meters
        '''
        elev = "http://ned.usgs.gov/epqs/pqs.php?x="+str(x[0])+"&y="+str(x[1])+"&units=Meters&output=xml"
        try:        
            response = urllib2.urlopen(elev)
            html = response.read()
            d = xmltodict.parse(html)
            g = float(d['USGS_Elevation_Point_Query_Service']['Elevation_Query']['Elevation'])
        except(BadStatusLine):
            try:        
                response = urllib2.urlopen(elev)
                html = response.read()
                d = xmltodict.parse(html)
                g = float(d['USGS_Elevation_Point_Query_Service']['Elevation_Query']['Elevation'])
            except(BadStatusLine):
                print "could not fetch %s" % html
                g = 0                
                pass                 
        return g
    
    @staticmethod     
    def USGSID(x):
        def dms(dec):
            DD = str(int(abs(dec)))
            MM = str(int((abs(dec) - int(DD))*60)).zfill(2)
            SS = str(int(round((((abs(dec) - int(DD))*60) - int(MM))*60, 0))).zfill(2)
            return DD+MM+SS
        return dms(x[1])+dms(x[0])+'01'
        
    def getInfo(self, html):
        '''
        Input
        -----
        html = location of data to be queried <http://waterservices.usgs.gov>         
        
        Output
        ------
        df = Pandas Dataframe containing data downloaded from USGS
        '''
        
        try:
            response = urllib2.urlopen(html)
            htmlresp = response.read()
            endhead = htmlresp.rfind('#\n')+2
            skip = htmlresp[:endhead].count('\n')
            skiplist = range(0,skip)
            skiplist.append(skip+1)
            endcom = htmlresp.find('\n#', endhead)
            if endcom <> -1:
                while endcom <= htmlresp.rfind('\n#'):
                
                    linenum = htmlresp[:endcom].count('\n')+1
                    endcomment = htmlresp.find('#\na',endcom)
                    commentnum = htmlresp[endcom:endcomment].count('\n')
                    skiplist = skiplist+range(linenum,commentnum+linenum+2)
                    endcom = htmlresp.find('\n#', endcomment)
                    if commentnum == 0:
                        break
            df = pd.read_table(html, sep="\t",skiprows=skiplist)#, comment='#')
            return df
        except(BadStatusLine):
            try:
                response = urllib2.urlopen(html)
                htmlresp = response.read()
                endhead = htmlresp.rfind('#\n')+2
                skip = htmlresp[:endhead].count('\n')
                skiplist = range(0,skip)
                skiplist.append(skip+1)
                endcom = htmlresp.find('\n#', endhead)
                if endcom <> -1:
                    while endcom <= htmlresp.rfind('\n#'):
                    
                        linenum = htmlresp[:endcom].count('\n')+1
                        endcomment = htmlresp.find('#\na',endcom)
                        commentnum = htmlresp[endcom:endcomment].count('\n')
                        skiplist = skiplist+range(linenum,commentnum+linenum+2)
                        endcom = htmlresp.find('\n#', endcomment)
                        if commentnum == 0:
                            break
                df = pd.read_table(html, sep="\t",skiprows=skiplist)#, comment='#')
                return df
            except(BadStatusLine):
                print "could not fetch %s" % html        
                pass            

    
    def parsesitelist(self, ListOfSites):
        '''
        Takes a list of USGS sites and turns it into a string format that can be used in the html REST format
        '''
        siteno = str(ListOfSites).replace(" ","")
        siteno = siteno.replace("]","")
        siteno = siteno.replace("[","")
        siteno = siteno.replace("','",",")
        siteno = siteno.replace("'","")
        siteno = siteno.replace('"',"")        
        return siteno
    
    def getStationInfo(self, sitenos):
        '''
        INPUT
        -----
        sitenos = list of usgs sites to get site info
        
        RETURNS
        -------
        siteinfo = pandas dataframe with station info
        '''
        siteno = self.parsesitelist(sitenos)        
        html = "http://waterservices.usgs.gov/nwis/site/?format=rdb&sites=" + siteno + "&siteOutput=expanded"
        siteinfo = self.getInfo(html)
        return siteinfo    
    
    def getStationInfoFromHUC(self, HUCS, sitetype=['GW'],datatype=['gw']):
        '''
        HUCS = list of HUCS to find sites
        sitetype = list of types of site you are searching for; options include 
            ST = Stream,SP = Spring,GW = Groundwater, SB = Other underground
            defaults to GW
        datatype = list of data types you are searching for; options include
            iv = instantaneous ,dv = daily values, sv = site visit, gw = groundwater level, qw = water quality, id =historical instantaneous
            defaults to gw
        RETURNS
        -------
        Pandas dataframe of sites in HUC
        '''
        sitetypes = self.parsesitelist(sitetype)
        datatypes = self.parsesitelist(datatype)
        HUC = self.parsesitelist(HUCS)
        stationhtml = "http://waterservices.usgs.gov/nwis/site/?format=rdb,1.0&huc=" + str(HUC) + "&siteType=" + sitetypes + "&hasDataTypeCd=" + datatypes
        siteinfo = self.getInfo(stationhtml)
        return siteinfo       
    
    def getStationsfromHUC(self, HUCS):
        HUC = self.parsesitelist(HUCS)
        stationhtml = "http://waterservices.usgs.gov/nwis/site/?format=rdb,1.0&huc=" + str(HUC) + "&siteType=GW&hasDataTypeCd=gw"
        sites = self.getInfo(stationhtml)
        stations = list(sites['site_no'].values)
        stations = [str(i) for i in stations]
        return stations
        
    def getWLfromHUC(self, HUCS):
        HUC = self.parsesitelist(HUCS)
        html = "http://waterservices.usgs.gov/nwis/gwlevels/?format=rdb&huc="+str(HUC)+"&startDT=1800-01-01&endDT="+str(datetime.today().year)+"-"+str(datetime.today().month).zfill(2)+"-"+str(datetime.today().day).zfill(2)
        wls = self.getInfo(html)
        return wls
        
    def getWLfromSite(self, sitenos):
        siteno = self.parsesitelist(sitenos)
        html = "http://waterservices.usgs.gov/nwis/gwlevels/?format=rdb&sites="+str(siteno)+"&startDT=1800-01-01&endDT="+str(datetime.today().year)+"-"+str(datetime.today().month).zfill(2)+"-"+str(datetime.today().day).zfill(2)
        wls = self.getInfo(html)
        return wls
    
    def getQfromSites(self, ListOfSites):
        '''
        get discharge data from site list
        '''
        siteno = self.parsesitelist(ListOfSites)
        html = "http://waterservices.usgs.gov/nwis/dv/?format=rdb&sites="+str(siteno)+"&parameterCd=00060"+"&startDT=1800-01-01&endDT="+str(datetime.today().year)+"-"+str(datetime.today().month).zfill(2)+"-"+str(datetime.today().day).zfill(2)
        wls = self.getInfo(html)
        wls.columns = ['agency_cd','site_no','datetime','discharge_cfs','quality_code']
        return wls
    
    def getQfromHUC(self, HUCS):
        '''
        get discharge data from HUC
        '''
        siteno = self.parsesitelist(HUCS)
        html = "http://waterservices.usgs.gov/nwis/dv/?format=rdb&huc="+str(HUC)+"&parameterCd=00060"+"&startDT=1800-01-01&endDT="+str(datetime.today().year)+"-"+str(datetime.today().month).zfill(2)+"-"+str(datetime.today().day).zfill(2)
        wls = self.getInfo(html)
        return wls
    
    def cleanGWL(self, data):
        '''
        Drops water level data of suspect quality based on lev_status_cd
        
        returns Pandas DataFrame
        '''
        CleanData = data[~data['lev_status_cd'].isin(['Z', 'R', 'V', 'P', 'O', 'F', 'W', 'G', 'S', 'C', 'E', 'N'])]
        return CleanData
    
    def WLStatdf(self, siteinfo, data):
        '''
        generates average water level statistics for a huc or list of hucs
        INPUT
        -----
        siteinfo = pandas dataframe of site information of nwis sites (made using a get station info function)
        data = pandas dataframe of data from nwis sites (made using a get station data function)
        
        RETURNS
        -------
        wlLongStatsGroups = pandas dataframe of standardized water levels over duration of measurement
        wlLongStatsGroups2 = pandas dataframe of change in average water levels over duration of measurement
        '''

        try:
            data.drop([u'agency_cd', u'site_tp_cd'], inplace=True, axis=1)
        except(ValueError):
            pass
        stationWL = pd.merge(data, siteinfo, on='site_no', how='left')

        stationWL['date'], stationWL['Year'], stationWL['Month'] = zip(*stationWL['lev_dt'].apply(lambda x: avgMeths.getyrmnth(x),1))
        stationWL.reset_index(inplace=True)
        stationWL.set_index('date',inplace=True)
        stationWL = self.cleanGWL(stationWL)
        # get averages by year, month, and site number
        grpstat = stationWL.groupby('site_no')['lev_va'].agg([np.std,np.mean,np.median, np.min,np.max,np.size]).reset_index()
        USGS_Site_Inf = stationWL.groupby('site_no')['lev_dt'].agg([np.min,np.max,np.size]).reset_index()
        USGS_Site_Info = USGS_Site_Inf[USGS_Site_Inf['size']>50]
        wlLong = stationWL[stationWL['site_no'].isin(list(USGS_Site_Info['site_no'].values))]
        wlLongStats = pd.merge(wlLong,grpstat, on='site_no', how='left')
        wlLongStats['stdWL'] = wlLongStats[['lev_va','mean','std']].apply(lambda x: avgMeths.stndrd(x),1 )
        wlLongStats['YRMO'] = wlLongStats[['Year','Month']].apply(lambda x: avgMeths.yrmo(x),1)
        wlLongStats['date'] = wlLongStats[['Year','Month']].apply(lambda x: avgMeths.adddate(x),1)
        self.wlMonthPlot = wlLongStats.groupby(['Month'])['stdWL'].mean().to_frame().plot()
        wlLongStats['levDiff'] = wlLongStats['lev_va'].diff()
            
        wlLongStatsGroups = wlLongStats.groupby(['date'])['stdWL'].agg({'mean':np.mean,'median':np.median,
                                                                        'standard':np.std, 'cnt':(lambda x: np.count_nonzero(~np.isnan(x))), 
                                                                        'err':(lambda x: 1.96*avgMeths.sumstats(x))})
        wlLongStatsGroups2 = wlLongStats.groupby(['date'])['levDiff'].agg({'mean':np.mean,'median':np.median, 'standard':np.std, 'cnt':(lambda x: np.count_nonzero(~np.isnan(x))), 'err':(lambda x: 1.96*avgMeths.sumstats(x))})
    
        wlLongStatsGroups['meanpluserr'] = wlLongStatsGroups['mean'] + wlLongStatsGroups['err']
        wlLongStatsGroups['meanminuserr'] = wlLongStatsGroups['mean'] - wlLongStatsGroups['err']
    
        wlLongStatsGroups2['meanpluserr'] = wlLongStatsGroups2['mean'] + wlLongStatsGroups2['err']
        wlLongStatsGroups2['meanminuserr'] = wlLongStatsGroups2['mean'] - wlLongStatsGroups2['err']
        
        return wlLongStatsGroups, wlLongStatsGroups2

    def HUCplot(self, siteinfo, data):
        '''
        Generates Statistics plots of NWIS WL data
        
        INPUT
        -----
        siteinfo = pandas dataframe of site information of nwis sites (made using a get station info function)
        data = pandas dataframe of data from nwis sites (made using a get station data function)
        
        RETURNS
        -------
        self.stand = standardized statistics
        self.diffs = difference statistics
        self.zPlot = seasonal variation plot
        self.wlPlot = plots of stadardized and differenced wl variations of duration of measurement
        '''
        df1,df2 = self.WLStatdf(siteinfo, data)
        wlLongSt = df1[df1['cnt']>2]
        wlLongSt2 = df2[df2['cnt']>2]
        
        self.stand = wlLongSt
        self.diffs = wlLongSt2
          
        fig1 = plt.figure()
        x = wlLongSt.index
        y = wlLongSt['mean']
        plt.plot(x,y,label='Average Groundwater Level Variation')
        plt.fill_between(wlLongSt.index, wlLongSt['meanpluserr'], wlLongSt['meanminuserr'], 
                         facecolor='blue', alpha=0.4, linewidth=0.5, label= "Std Error")
        plt.grid(which='both')
        plt.ylabel('Depth to Water z-score')
        plt.xticks(rotation=45)
        self.zPlot = fig1        
        
        fig2 = plt.figure()
        x = wlLongSt2.index
        y = wlLongSt2['mean']
        plt.plot(x,y,label='Average Groundwater Level Changes')
        plt.fill_between(wlLongSt.index, wlLongSt2['meanpluserr'], wlLongSt2['meanminuserr'], 
                         facecolor='blue', alpha=0.4, linewidth=0.5, label= "Std Error")
        plt.grid(which='both')
        plt.ylabel('Change in Average Depth to Water (ft)')
        plt.xticks(rotation=45)
        self.wlPlot = fig2
        
        return fig1, fig2, wlLongSt, wlLongSt2
