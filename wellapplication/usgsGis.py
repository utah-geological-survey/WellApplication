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
import requests
import json

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
    
    def get_nwis(self, val_list, selectType ='dv_site', start_date = '1800-01-01', end_date = ''):
        """
        Request stream gauge data from the USGS NWIS.
        Args:
            val_list (str or int or list of either):
                can be a list or 8-digit hucs or stations or single 
                a valid site is 01585200; a valid huc is 16010204
            selectType (str):
                options: 'dv_site','dv_huc','gw_site',or 'gw_huc'
            start_date (str):
               should take on the form yyyy-mm-dd; default is '1800-01-01'
            end_date (str):
                should take on the form yyyy-mm-dd; default is today
        Returns:
            a station and a data Pandas dataframe.
        Raises:
            ConnectionError  due to connection problems like refused connection
            or DNS Error.

        Example::
            >>> import wellapplication as wa
            >>> USGS = wa.usgs()
            >>> site, data = USGS.get_nwis('01585200', 'dv_site', '2012-06-01', '2012-07-01')

        The specification for this service is located here:
        http://waterservices.usgs.gov/rest/IV-Service.html
        
        This function was adapted from: https://github.com/mroberge/hydrofunctions
        """
        val_list = self.parsesitelist(val_list)

        if end_date == '':
            dy = datetime.today()
            end_date = str(dy.year)+'-'+str(dy.month)+'-'+str(dy.day)

        header = {'Accept-encoding': 'gzip'}

        
        valdict = {
            'dv_site':{'format': 'json', 'sites': val_list, 'parameterCd': '00060',
                      'startDT': start_date, 'endDT': end_date},
            'dv_huc':{'format': 'json', 'huc': val_list, 'parameterCd': '00060',
                      'startDT': start_date, 'endDT': end_date},
            'gw_site':{'format': 'json', 'sites': val_list, 'siteType':'GW','siteStatus':'all',
                      'startDT': start_date, 'endDT': end_date},
            'gw_huc':{'format': 'json', 'huc': val_list, 'siteType':'GW','siteStatus':'all',
                      'startDT': start_date, 'endDT': end_date}
            }


        url = 'http://waterservices.usgs.gov/nwis/'
            
        if selectType == 'dv_site' or selectType == 'dv_huc':
            service = 'dv'
        elif selectType == 'gw_site' or selectType == 'gw_huc':
            service = 'gwlevels'
        url = url + service + '/?'
        response_ob = requests.get(url, params=valdict[selectType], headers=header)

        nwis_dict = response_ob.json()

        dt = nwis_dict['value']['timeSeries']

        station_id,lat,lon,srs,station_type,station_nm =  [],[],[],[],[],[]
        f = {}
        for i in range(len(dt)):

            station_id.append(dt[i]['sourceInfo']['siteCode'][0]['value'])
            lat.append(dt[i]['sourceInfo']['geoLocation'][u'geogLocation']['latitude'])
            lon.append(dt[i]['sourceInfo']['geoLocation'][u'geogLocation']['longitude'])
            srs.append(dt[i]['sourceInfo']['geoLocation'][u'geogLocation']['srs'])
            station_type.append(dt[i]['sourceInfo']['siteProperty'][0]['value'])
            station_nm .append(dt[i]['sourceInfo'][ u'siteName'])

            df = pd.DataFrame(dt[i]['values'][0]['value'], columns=['dateTime', 'value'])
            df.index = pd.to_datetime(df.pop('dateTime'))
            df.value = df.value.astype(float)
            df.index.name = 'datetime'
            df.replace(to_replace='-999999', value=np.nan)
            f[dt[i]['sourceInfo']['siteCode'][0]['value']] = df
        stat_dict = {'station_id':station_id,'lat':lat,'lon':lon,'srs':srs,'station_nm':station_nm,'stat_type':station_type}
        stations = pd.DataFrame(stat_dict)
        if len(dt) > 1:
            data = pd.concat(f)
        else:
            data = f[dt[0]['sourceInfo']['siteCode'][0]['value']]
            data['station_id'] = dt[0]['sourceInfo']['siteCode'][0]['value']
        return stations,data
    
    
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
            linefile = urllib2.urlopen(html).readlines()
            numlist=[]
            num=0
            for line in linefile:
                if line.startswith("#"):
                    numlist.append(num)
                num += 1
            numlist.append(numlist[-1]+2)
            df = pd.read_table(html, sep="\t",skiprows=numlist)#, comment='#')
            return df
        except(BadStatusLine):
            try:
                linefile = urllib2.urlopen(html).readlines()
                numlist=[]
                num=0
                for line in linefile:
                    if line.startswith("#"):
                        numlist.append(num)
                    num += 1
                numlist.append(numlist[-1]+2)
                df = pd.read_table(html, sep="\t",skiprows=numlist)#, comment='#')
                return df
            except(BadStatusLine):
                print "could not fetch %s" % html        
                pass            

    def getQInfo(self,html):
        '''
        Input
        -----
        html = location of data to be queried <http://waterservices.usgs.gov>         
    
        Output
        ------
        df = Pandas Dataframe containing data downloaded from USGS
        '''
        linefile = urllib2.urlopen(html).readlines()
        numlist=[]
        num=0
        for line in linefile:
            if line.startswith("#"):
                numlist.append(num)
            elif line.startswith("5s"):
                numlist.append(num)
            num += 1
        df = pd.read_table(html, sep="\t",skiprows=numlist, index_col='datetime',parse_dates=True)
        df.columns = ['agency','site','discharge','qual']
        df['discharge'] = pd.to_numeric(df['discharge'], errors='coerce')
        return df
    
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
        html = "http://waterservices.usgs.gov/nwis/dv/?format=rdb&sites={0}&parameterCd=00060&startDT=1800-01-01&endDT={1}".format(siteno,datetime.today())
        wls = self.getQInfo(html)
        return wls
    
    def getQfromHUC(self, HUCS):
        '''
        get discharge data from HUC
        '''
        siteno = self.parsesitelist(HUCS)
        html = "http://waterservices.usgs.gov/nwis/dv/?format=rdb&parameterCd=00060&huc={0}&startDT=1800-01-01&endDT={1:%Y-%m-%d}".format(HUCS,datetime.today())
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
        
    def get_huc(self, x):

        """
        Receive the content of ``url``, parse it as JSON and return the object.
    
        Parameters
        ----------
        x = [longitude, latitude]
    
        Returns
        -------
        HUC12, HUC12_Name
        """
        values = {
            'geometry': '-111.406,40.3499',
            'geometryType':'esriGeometryPoint',
            'inSR':'4326',
            'spatialRel':'esriSpatialRelIntersects',
            'returnGeometry':'false',
            'outFields':'HUC12,Name',
            'returnDistinctValues':'true',
            'f':'pjson'
            }


        huc_url = 'https://services.nationalmap.gov/arcgis/rest/services/nhd/mapserver/8/query?'
        response = requests.get(huc_url,params=values).json()
        response['features'][0]['attributes']['HUC12'], response['features'][0]['attributes']['NAME']
