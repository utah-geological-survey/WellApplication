# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import os

from .transport import *
from .usgs import *
from .chem import *
from .mesopy import *
from .graphs import *
from .MannKendall import *

rootpath = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
version_file = open(os.path.join(rootpath, 'VERSION'))
version = version_file.read().strip()

__version__ = version
__author__ = 'Paul Inkenbrandt'
__name__ = 'wellapplication'

__all__ = ['WQP',
           'get_response',
           'get_wqp_stations',
           'get_wqp_results',
           'massage_results',
           'datetimefix',
           'parnorm',
           'unitfix',
           'massage_stations',
           'piv_chem']
