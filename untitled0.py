# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 12:13:28 2022

@author: cosbo
"""

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
import matplotlib
import matplotlib.pyplot as plt

center_coord = SkyCoord('02h21m00s +57d07m42s')
vizier = Vizier(
    columns=['RAJ2000', 'DEJ2000', 'pmRA', 'pmDE'],
    column_filters={'BPmag': '<16', 'pmRA': '!=', 'pmDE': '!='}, # число больше — звёзд больше
    row_limit=10000
)
stars = vizier.query_region(
    center_coord,
    width=1.0 * u.deg,
    height=1.0 * u.deg,
    catalog=['I/350'],  # Gaia EDR3
)[0]

ra = stars['RAJ2000']._data   # прямое восхождение, аналог долготы
dec = stars['DEJ2000']._data  # склонение, аналог широты
x1 = (ra - ra.mean()) * np.cos(dec / 180 * np.pi) + ra.mean()
x2 = dec
v1 = stars['pmRA']._data
v2 = stars['pmDE']._data
plt.figure()
plt.scatter(x1, x2, s=0.25)

