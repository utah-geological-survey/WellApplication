# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 13:03:00 2016

@author: p
"""

import wellapplication as wa



def test_getelev():
    x = [-111.21,41.4]
    g = wa.getelev(x)
    assert g > 100.0

def test_gethuc():
    x = [-111.21,41.4]
    g = wa.get_huc(x)
    assert len(g[0])>0
