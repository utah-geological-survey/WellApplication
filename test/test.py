# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 13:03:00 2016

@author: p
"""

import wellapplication as wa
import pandas as pd
import matplotlib
import numpy as np

m = wa.Meso(token='demotoken')

def test_getelev():
    print('Testing getelev')
    x = [-111.21, 41.4]
    m = wa.getelev(x)
    assert m > 100.0

def test_gethuc():
    print('Testing gethuc')
    x = [-111.21, 41.4]
    huc_data = wa.get_huc(x)
    assert len(huc_data[0])>0

def test_USGSID():
    print('Testing USGSID')
    x = [-111.21, 41.4]
    usgs_id = wa.USGSID(x)
    assert usgs_id == '412400111123601'

def test_nwis():
    nw = wa.nwis('dv', '01585200', 'sites')
    assert len(nw.sites) == 1
    
def test_nwis_gw():
    nw = wa.nwis('gwlevels','16010204','huc',siteStatus='all')
    df = nw.avg_wl()
    assert len(df) > 5

def test_fdc():
    d16 = wa.nwis('dv','01659500','sites')
    ci = wa.fdc(d16.data,'value',1900,2016)
    assert type(ci[0]) == list 

def test_mktest():
    x = range(0,100)
    trend = wa.MannKendall.mk_test(x,0.05)
    assert trend.trend == 'increasing'

def test_pipe():
    Chem =  {'Type':[1,2,2,3], 'Cl':[1.72,0.90,4.09,1.52], 'HCO3':[4.02,1.28,4.29,3.04], 
             'SO4':[0.58,0.54,0.38,0.46], 'NaK':[1.40,0.90,3.38,2.86], 'Ca':[4.53,None,4.74,1.90], 
             'Mg':[0.79,0.74,0.72,0.66], 'EC':[672.0,308.0,884.0,542.0], 'NO3':[0.4,0.36,0.08,0.40], 
             'Sicc':[0.21,0.56,None,-0.41]}  
    chem = pd.DataFrame(Chem)
    pipr = wa.piper(chem)
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
    
def test_WQP():
    wqq = wa.WQP('16010204','huc')
    wqq.massage_results()
    pivchem = wqq.piv_chem()
    assert 'Alk' in pivchem.columns

def test_WQ2():
    wqq = wa.WQP('16010204','huc')
    wqq.massage_stations()
    df = wqq.stations
    assert "OrgId" in list(df.columns)
    
def test_imp_new_well():
    inputfile = "docs/ag13c 2016-08-02.xle"
    manualwls = "docs/All tape measurements.csv"
    manual = pd.read_csv(manualwls, index_col="DateTime", engine="python")
    barofile = "docs/baro.csv"
    baro = pd.read_csv(barofile,index_col=0, parse_dates=True)
    wellinfo = pd.read_csv("docs/wellinfo4.csv")
    g, drift, wellname = wa.imp_new_well(inputfile, wellinfo, manual, baro)
    assert wellname == 'ag13c'
    
def test_well_baro_merge():
    inputfile = "docs/ag13c 2016-08-02.xle"
    manualwls = "docs/All tape measurements.csv"
    xle = "docs/ag13c 2016-08-02.xle"
    xle_df = wa.new_xle_imp(xle)
    manual = pd.read_csv(manualwls, index_col="DateTime", engine="python")
    barofile = "docs/baro.csv"
    baro = pd.read_csv(barofile,index_col=0, parse_dates=True)
    baro['Level'] = baro['pw03']
    wellinfo = pd.read_csv("docs/wellinfo4.csv")
    assert len(wa.well_baro_merge(xle_df, baro, sampint=60)) > 10

def test_fix_drift():
    xle = "docs/ag13c 2016-08-02.xle"
    xle_df = wa.new_xle_imp(xle)
    
    manualwls = "docs/All tape measurements.csv"
    manual = pd.read_csv(manualwls, index_col="DateTime", engine="python")
    manual35 = manual[manual['WellID']==35]
    manual35['dt'] = pd.to_datetime(manual35.index)
    manual_35 = manual35.reset_index()
    manual_35.set_index('dt',inplace=True)
    fd = wa.fix_drift(xle_df, manual_35, meas='Level', manmeas='MeasuredDTW', outcolname='DriftCorrection')
    assert 'DriftCorrection' in list(fd[0].columns)
    
def test_getwellid():
    inputfile = "docs/ag13c 2016-08-02.xle"
    wellinfo = pd.read_csv("docs/wellinfo4.csv")
    wid = wa.getwellid(inputfile, wellinfo)
    assert wid[1] == 35

def test_barodistance():
    wellinfo = pd.read_csv("docs/wellinfo4.csv")
    bd = wa.barodistance(wellinfo)
    assert 'closest_baro' in list(bd.columns)

def test_imp_new_well_csv():
    inputfile = "docs/ag14a 2016-08-02.csv"
    manualwls = "docs/All tape measurements.csv"
    manual = pd.read_csv(manualwls, index_col="DateTime", engine="python")
    barofile = "docs/baro.csv"
    baro = pd.read_csv(barofile,index_col=0, parse_dates=True)
    wellinfo = pd.read_csv("docs/wellinfo4.csv")
    g, drift, wellname = wa.imp_new_well(inputfile, wellinfo, manual, baro)
    assert wellname == 'ag14a'
    
def test_jumpfix():
    xle = "docs/ag13c 2016-08-02.xle"
    df = wa.new_xle_imp(xle)
    jf = wa.jumpfix(df, 'Level', threashold=0.005)
    assert jf['newVal'][-1] > 10

def test_gantt():
    ashley = wa.nwis('dv', '09265500', 'sites')
    gn = wa.gantt(ashley.data, stations=['value'])
    assert type(gn.gantt()[2]) == matplotlib.figure.Figure

def test_scatterColor():
    x = np.arange(1, 100, 1)
    y = np.arange(0.1, 10.0, 0.1)
    w = np.arange(5, 500, 5)
    out = wa.scatterColor(x, y, w)
    assert round(out[0], 1) == 0.1

def test_get_info():
    nw = wa.nwis('gwlevels', '16010204', 'huc', siteStatus='all')
    df = nw.get_info(siteStatus='all')
    assert 'site_no' in list(df.columns)
