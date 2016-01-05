# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 00:30:36 2016

@author: p
"""
import urllib2
import xmltodict
import pandas as pd

class usgsGis:
    
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
        response = urllib2.urlopen(elev)
        html = response.read()
        d = xmltodict.parse(html)
        return float(d['USGS_Elevation_Point_Query_Service']['Elevation_Query']['Elevation'])
        
        
    def USGSID(x):
        def dms(dec):
            DD = str(int(abs(dec)))
            MM = str(int((abs(dec) - int(DD))*60)).zfill(2)
            SS = str(int(round((((abs(dec) - int(DD))*60) - int(MM))*60, 0))).zfill(2)
            return DD+MM+SS
        return dms(x[1])+dms(x[0])+'01'
        
    def getInfo(html):
        '''
        Input
        -----
        html = location of data to be queried <http://waterservices.usgs.gov>         
        
        Output
        ------
        Pandas Dataframe containing data downloaded from USGS
        '''
        response = urllib2.urlopen(html)
        html = response.read()
        skip = html[:html.rfind('#\n')+2].count('\n')
        skiplist = range(0,skip)
        skiplist.append(skip+1)
        df = pd.read_table(html, sep="\t",skiprows=skiplist)
        return df