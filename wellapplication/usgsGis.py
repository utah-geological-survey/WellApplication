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
import numpy as np
import matplotlib.pyplot as plt
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
            skip = htmlresp[:htmlresp.rfind('#\n')+2].count('\n')
            skiplist = range(0,skip)
            skiplist.append(skip+1)
            df = pd.read_table(html, sep="\t",skiprows=skiplist)
            return df
        except(BadStatusLine):
            try:
                response = urllib2.urlopen(html)
                htmlresp = response.read()
                skip = htmlresp[:htmlresp.rfind('#\n')+2].count('\n')
                skiplist = range(0,skip)
                skiplist.append(skip+1)
                df = pd.read_table(html, sep="\t",skiprows=skiplist)
                return df
            except(BadStatusLine):
                print "could not fetch %s" % html        
                pass            

        
    def getStationInfo(self, siteno):
        html = "http://waterservices.usgs.gov/nwis/site/?format=rdb&sites=" + siteno + "&siteOutput=expanded"
        siteinfo = self.getInfo(html)
        return siteinfo
    
    def parsesitelist(self, ListOfSites):
        siteno = str(ListOfSites).replace(" ","")
        siteno = siteno.replace("]","")
        siteno = siteno.replace("[","")
        siteno = siteno.replace("','",",")
        siteno = siteno.replace("'","")
        return siteno
    
    def getStationInfoFromList(self, ListOfSites):
        siteno = self.parsesitelist(ListOfSites)
        html = "http://waterservices.usgs.gov/nwis/site/?format=rdb&sites=" + siteno + "&siteOutput=expanded"
        siteinfo = self.getInfo(html)
        return siteinfo

    def getStationInfoFromHUC(self, HUC):
        stationhtml = "http://waterservices.usgs.gov/nwis/site/?format=rdb,1.0&huc=" + str(HUC) + "&siteType=GW&hasDataTypeCd=gw"
        siteinfo = self.getInfo(stationhtml)
        return siteinfo       
    
    def getStationsfromHUC(self, HUC):
        stationhtml = "http://waterservices.usgs.gov/nwis/site/?format=rdb,1.0&huc=" + str(HUC) + "&siteType=GW&hasDataTypeCd=gw"
        sites = self.getInfo(stationhtml)
        stations = list(sites['site_no'].values)
        stations = [str(i) for i in stations]
        return stations
        
    def getWLfromHUC(self, HUC):
        html = "http://waterservices.usgs.gov/nwis/gwlevels/?format=rdb&huc="+str(HUC)+"&startDT=1800-01-01&endDT="+str(datetime.today().year)+"-"+str(datetime.today().month).zfill(2)+"-"+str(datetime.today().day).zfill(2)
        wls = self.getInfo(html)
        return wls
        
    def getWLfromSite(self, siteno):
        html = "http://waterservices.usgs.gov/nwis/gwlevels/?format=rdb&sites="+str(siteno)+"&startDT=1800-01-01&endDT="+str(datetime.today().year)+"-"+str(datetime.today().month).zfill(2)+"-"+str(datetime.today().day).zfill(2)
        wls = self.getInfo(html)
        return wls
    
    def getWLfromSiteList(self, ListOfSites):
        siteno = self.parsesitelist(ListOfSites)
        html = "http://waterservices.usgs.gov/nwis/gwlevels/?format=rdb&sites="+str(siteno)+"&startDT=1800-01-01&endDT="+str(datetime.today().year)+"-"+str(datetime.today().month).zfill(2)+"-"+str(datetime.today().day).zfill(2)
        wls = self.getInfo(html)
        return wls
    
    def hucPlot(self, HUC):
        #stations = USGS.getStationsfromHUC(str(HUC))
        siteinfo = self.getStationInfoFromHUC(str(HUC))
        data = self.getWLfromHUC(HUC)
        data.drop([u'agency_cd', u'site_tp_cd'], inplace=True, axis=1)
        stationWL = pd.merge(data, siteinfo, on='site_no', how='left')
        stationWL['date'], stationWL['Year'], stationWL['Month'] = zip(*stationWL['lev_dt'].apply(lambda x: avgMeths.getyrmnth(x),1))
        stationWL.reset_index(inplace=True)
        stationWL.set_index('date',inplace=True)
        # get averages by year, month, and site number
        grpstat = stationWL.groupby('site_no')['lev_va'].agg([np.std,np.mean,np.median, np.min,np.max,np.size]).reset_index()
        USGS_Site_Inf = stationWL.groupby('site_no')['lev_dt'].agg([np.min,np.max,np.size]).reset_index()
        USGS_Site_Info = USGS_Site_Inf[USGS_Site_Inf['size']>50]
        wlLong = stationWL[stationWL['site_no'].isin(list(USGS_Site_Info['site_no'].values))]
        wlLongStats = pd.merge(wlLong,grpstat, on='site_no', how='left')
        wlLongStats['stdWL'] = wlLongStats[['lev_va','mean','std']].apply(lambda x: avgMeths.stndrd(x),1 )
        wlLongStats['YRMO'] = wlLongStats[['Year','Month']].apply(lambda x: avgMeths.yrmo(x),1)
        wlLongStats['date'] = wlLongStats[['Year','Month']].apply(lambda x: avgMeths.adddate(x),1)
        wlLongStats.groupby(['Month'])['stdWL'].mean().to_frame().plot()
        wlLongStats['levDiff'] = wlLongStats['lev_va'].diff()
            
        wlLongStatsGroups = wlLongStats.groupby(['date'])['stdWL'].agg({'mean':np.mean,'median':np.median,
                                                                        'standard':np.std, 
                                                                        'cnt':(lambda x: np.count_nonzero(~np.isnan(x))), 
                                                                        'err':(lambda x: 1.96*np.std(x)/np.count_nonzero(~np.isnan(x)))})
        wlLongStatsGroups2 = wlLongStats.groupby(['date'])['levDiff'].agg({'mean':np.mean,'median':np.median, 'standard':np.std, 'cnt':(lambda x: np.count_nonzero(~np.isnan(x))), 'err':(lambda x: 1.96*np.std(x)/np.count_nonzero(~np.isnan(x)))})
    
        wlLongStatsGroups['meanpluserr'] = wlLongStatsGroups['mean'] + wlLongStatsGroups['err']
        wlLongStatsGroups['meanminuserr'] = wlLongStatsGroups['mean'] - wlLongStatsGroups['err']
    
        wlLongStatsGroups2['meanpluserr'] = wlLongStatsGroups2['mean'] + wlLongStatsGroups2['err']
        wlLongStatsGroups2['meanminuserr'] = wlLongStatsGroups2['mean'] - wlLongStatsGroups2['err']
        
        wlLongSt = wlLongStatsGroups[wlLongStatsGroups['cnt']>2]
        wlLongSt2 = wlLongStatsGroups2[wlLongStatsGroups2['cnt']>2]
        
        plt.figure()
        x = wlLongSt.index
        y = wlLongSt['mean']
        plt.plot(x,y,label='Average Groundwater Level Variation')
        plt.fill_between(wlLongSt.index, wlLongSt['meanpluserr'], wlLongSt['meanminuserr'], 
                         facecolor='blue', alpha=0.4, linewidth=0.5, label= "Std Error")
        plt.grid(which='both')
        plt.ylabel('Depth to Water z-score')
        plt.xticks(rotation=45)
        
        plt.figure()
        x = wlLongSt2.index
        y = wlLongSt2['mean']
        plt.plot(x,y,label='Average Groundwater Level Changes')
        plt.fill_between(wlLongSt.index, wlLongSt2['meanpluserr'], wlLongSt2['meanminuserr'], 
                         facecolor='blue', alpha=0.4, linewidth=0.5, label= "Std Error")
        plt.grid(which='both')
        plt.ylabel('Change in Average Depth to Water (ft)')
        plt.xticks(rotation=45)

    