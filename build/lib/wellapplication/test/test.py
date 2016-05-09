# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 13:03:00 2016

@author: p
"""

import unittest
import wellapplication as wa



class TestStringMethods(unittest.TestCase):

    def __init__(self):
        USGS = wa.usgs()
        return USGS
        
    def test_USGSdf(self):
        USGS.HUCdf(16020301)
        self.assertEqual(USGS.wlMonthPlot, matplotlib.axes._subplots.AxesSubplot)
            

if __name__ == '__main__':
    unittest.main()