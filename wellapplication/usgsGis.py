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


    