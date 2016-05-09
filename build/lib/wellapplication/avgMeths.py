# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 22:31:26 2016

@author: p
"""
import numpy as np
from datetime import datetime

def stndrd(x):
    if np.isfinite(x[2]):
        return (x[0]-x[1])/x[2]
    else:
        return x[0]
        
def yrmo(x):
    return int(str(int(x[0]))+str(int(x[1])).zfill(2))
    
def adddate(x):
    return datetime(x[0],x[1],15)
    
def getyrmnth(x):
    dt = datetime.strptime(x, '%Y-%m-%d')
    return dt, dt.year, dt.month
    
def proj(x):
    from pyproj import Proj, transform
    inProj = Proj(init='epsg:4326') #WGS84
    outProj = Proj(init='epsg:2152') #NAD83(CSRS98) / UTM zone 12N
    x2,y2 = transform(inProj,outProj,x[0],x[1])
    return x2, y2

def projy(x):
    from pyproj import Proj, transform
    inProj = Proj(init='epsg:4326') #WGS84
    outProj = Proj(init='epsg:2152') #NAD83(CSRS98) / UTM zone 12N
    x2,y2 = transform(inProj,outProj,x[0],x[1])
    return y2

def projx(x):
    from pyproj import Proj, transform
    inProj = Proj(proj='latlong', datum='WGS84')#, init='epsg:4326') #WGS84
    outProj = Proj(proj='utm', zone=12, ellps='WGS84')#, init='epsg:2152') #NAD83(CSRS98) / UTM zone 12N
    x2,y2 = transform(inProj,outProj,x[0],x[1], radians=False)
    return x2
    
def getwlelev(x):
    return x[1] - (x[0]/3.2808)
    
def sumstats(x):
    if np.count_nonzero(~np.isnan(x)) == 0:
        return 1
    else:
        return np.std(x)/np.count_nonzero(~np.isnan(x))
    
