#!/usr/bin/env python3

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from mixfit import em_double_cluster, T
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
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


    x = np.asarray([x1, x2, v1, v2]).T
    mu = np.asarray([1, 2, 3, 4])
    mu = np.broadcast_to(mu, (4345, 4))
    sigmavec = [0.01, 0.01, 0.01, 0.2]
    theta0 = (0.2, 0.3, [-10, -2], [34.85, 57.17], [35.3, 57.17],
              [0, 0, 0.5, 0.5], [0.5, 0.5], [0.5, 0.5])
    r = em_double_cluster(x, *theta0)
    T1, T2, T3 = T(x, *r)
    
    xx = pd.DataFrame({'x1': x1, 'x2': x2, 'value': T1+T2})
    print(r[0]+r[1])
    plt.figure()
    plt.scatter(xx.x1, xx.x2, c=xx.value, cmap="Oranges")
    plt.plot(r[3][0], r[3][1], 'o', ms=4, color='blue')
    plt.plot(r[4][0], r[4][1], 'o', ms=4, color='blue')
    mu1vec = np.asarray([[-10,-2], [34.85, 57.17]]).flatten()
