# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 13:03:00 2016

@author: p
"""

import unittest
import wellapplication as wa



class TestMethods(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_getelev(self):
        x = [-111.21,41.4]
        g = wa.getelev(x)
        self.assertTrue(g > 100.0)

if __name__ == '__main__':
    unittest.main()
