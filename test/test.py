# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 13:03:00 2016

@author: p
"""

import wellapplication as wa
import pandas as pd
import matplotlib
from MesoPy import Meso, MesoPyError

m = Meso(token='demotoken')

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
    ok_(var_list)


def testmetadata():
    stations = m.metadata(radius=['wbb', 5])
    ok_(stations)


def testtimeseries():
    timeseries = m.timeseries(stid='kfnl', start='201504261800', end='201504262300')
    ok_(timeseries)


def testclimatology():
    climatology = m.climatology(stid='kden', startclim='04260000', endclim='04270000', units='precip|in')
    ok_(climatology)


def testprecip():
    precip = m.precip(stid=['kfnl', 'ksdf'], start='201504261800', end='201504271200', units='precip|in')
    ok_(precip)

def testclimatestats():
    climate_stats = m.climate_stats(stid='mtmet', startclim='03240000', endclim='03280000', type='all')
    ok_(climate_stats)

def testtimestats():
    stats = m.time_stats(stid='mtmet', start='201403240000', end='201403280000', type='all')
    ok_(stats)

def testlatency():
    latency = m.latency(stid='mtmet', start='201403240000', end='201403280000')
    ok_(latency)

def testnettypes():
    nettypes = m.networktypes()
    ok_(nettypes)

def testnetworks():
    networks = m.networks()
    ok_(networks)

def testattime():
    attime = m.attime(stid='kfnl', attime='201504261800', within='30')
    ok_(attime)

# Miscellaneous Tests

def testlateststrlist():
    latest = m.latest(stid=['kfnl', 'kden', 'ksdf'], within='90')
    print(latest)
    eq_(len(latest['STATION']), 3)

# Error Handling
@raises(MesoPyError)
def testbadurlstring():
    latest = m.latest(stid='')
    print(latest)


@raises(MesoPyError)
def testauth():
    badtoken = Meso(token='3030')
    badtoken.latest(stid=['kfnl', 'kden', 'ksdf'], within='30')


@raises(MesoPyError)
def testgeoparms():
    m.precip(start='201504261800', end='201504271200', units='precip|in')
