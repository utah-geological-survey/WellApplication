# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 13:03:00 2016

@author: p
"""

import wellapplication as wa
import pandas as pd
import matplotlib


m = wa.Meso(token='demotoken')

def test_getelev():
    x = [-111.21, 41.4]
    m = wa.getelev(x)
    assert m > 100.0

def test_gethuc():
    x = [-111.21, 41.4]
    huc_data = wa.get_huc(x)
    assert len(huc_data[0])>0

def test_USGSID():
    x = [-111.21, 41.4]
    usgs_id = wa.USGSID(x)
    assert usgs_id == '412400111123601'

def test_nwis():
    nw = wa.nwis('dv', '01585200', 'sites')
    assert len(nw.sites) == 1
    
def test_nwis_gw():
    nw = wa.nwis('gwlevels','16010204','huc',siteStatus='all')
    df = nw.cleanGWL(nw.data)
    assert len(df) > 5
    
def test_mktest():
    x = range(0,100)
    trend = wa.MannKendall.mk_test(x,0.05)
    assert trend.trend == 'increasing'

def test_pipe():
    pipr = wa.piper()
    Chem =  {'Type':[1,2,2,3], 'Cl':[1.72,0.90,4.09,1.52], 'HCO3':[4.02,1.28,4.29,3.04], 
             'SO4':[0.58,0.54,0.38,0.46], 'NaK':[1.40,0.90,3.38,2.86], 'Ca':[4.53,None,4.74,1.90], 
             'Mg':[0.79,0.74,0.72,0.66], 'EC':[672.0,308.0,884.0,542.0], 'NO3':[0.4,0.36,0.08,0.40], 
             'Sicc':[0.21,0.56,None,-0.41]}  
    chem = pd.DataFrame(Chem)
    pipr.piperplot(chem)
    assert type(pipr.plot) == matplotlib.figure.Figure
    
def test_new_xle_imp():
    xle = 'docs/20160919_LittleHobble.xle'
    xle_df = wa.new_xle_imp(xle)
    assert len(xle_df) > 0 

def test_xle_head_table():
    xle_dir = 'docs/'
    dir_df = wa.xle_head_table(xle_dir)
    assert len(xle_dir) > 0

def test_dataendclean():
    xle = 'docs/20160919_LittleHobble.xle'
    df = wa.new_xle_imp(xle)
    x = 'Level'
    xle1 = wa.dataendclean(df, x)
    assert len(xle1) > 1
    
def test_smoother():
    xle = 'docs/20160919_LittleHobble.xle'
    df = wa.new_xle_imp(xle)
    x = 'Level'
    xle1 = wa.smoother(df, x, sd=1)
    assert len(xle1) > 1
    
def test_hourly_resample():
    xle = 'docs/20160919_LittleHobble.xle'
    df = wa.new_xle_imp(xle)
    xle1 = wa.hourly_resample(df, minutes=30)

# Basic Function Tests
def testvars():
    var_list = m.variables()

def testmetadata():
    stations = m.metadata(radius=['wbb', 5])
    
