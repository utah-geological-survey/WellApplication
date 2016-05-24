# -*- coding: utf-8 -*-

__version__ = '0.2.12'
__author__ = 'Paul Inkenbrandt'
__name__ = 'wellapplication'

from transport import transport
from usgsGis import usgs
from chem import WQP
from graphs import piper, fdc, gantt
import MannKendall
import avgMeths
