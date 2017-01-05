# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 13:03:00 2016

@author: p
"""

import wellapplication as wa
import matplotlib
import pandas as pd



x = [-111.21,41.4]
val_list = '01585200'
val_huc = '16010204'
stat, levs = wa.get_nwis(val_huc, 'gw_huc')
g = wa.get_nwis(val_list, 'dv_site', '2012-06-01', '2012-07-01')
f = wa.avg_wl('16030006',numObs= 50, grptype = 'monthly', avgtype = 'avgDiffWL')
        
def test_getelev():
    m = wa.getelev(x)
    assert m > 100.0

def test_gethuc(x):
    huc_data = wa.get_huc(x)
    assert len(huc_data[0])>0

def test_USGSID(x):
    usgs_id = wa.USGSID([-111.21,41.4])
    assert usgs_id == '412400111123601'

def test_get_station_info():
    assert len(wa.get_station_info(['01585200','10136500'])) == 2
    
def test_cleanGWL(levs):
    levs2 = wa.cleanGWL(levs)
    assert type(levs2.qualifiers[0]) == str

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
    
def test_fdc():
    sites,levs = wa.get_nwis('01585200')
    assert len(wa.fdc(levs,'value')[0]) > 100

def test_reccur():
    cession = wa.graphs.recess()
    df = wa.get_nwis('01585200',selectType='dv_site', start_date='1968-01-01', end_date='1968-06-01')[1]
    type(cession.recession(df,'value',[1968,1,15],[1968,1,20])[1]) == pd.indexes.numeric.Float64Index
