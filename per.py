#!/usr/bin/env python3

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
from mixfit import em_double_cluster, T
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import AutoMinorLocator
import json


style_default = matplotlib.font_manager.FontProperties()
style_default.set_size('large')
style_default.set_family(['Calibri', 'Helvetica', 'Arial', 'serif'])
style_BIG = matplotlib.font_manager.FontProperties()
style_BIG.set_size('xx-large')
style_BIG.set_family(['Calibri', 'Helvetica', 'Arial', 'serif'])


def add_colorbar(mappable, side, label='probability of being in a cluster',
                 style=style_default):

    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(side, size="5%", pad=0.6)
    if side == 'right' or side == 'left':
        cbar = fig.colorbar(mappable, cax=cax, orientation='vertical')
        cax.yaxis.set_ticks_position(side)
        cbar.ax.set_ylabel(label)
    else:
        cbar = fig.colorbar(mappable, cax=cax, orientation='horizontal')
        cax.xaxis.set_ticks_position(side)
        cbar.ax.set_xlabel(label, fontproperties=style)
    plt.sca(last_axes)
    return cbar


if __name__ == "__main__":
    center_coord = SkyCoord('02h21m00s +57d07m42s')
    vizier = Vizier(
        columns=['RAJ2000', 'DEJ2000', 'pmRA', 'pmDE'],
        column_filters={'BPmag': '<16', 'pmRA': '!=', 'pmDE': '!='},
        # число больше — звёзд больше
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

    theta0 = (0.2, 0.3, [-10, -2], [34.85, 57.17], [35.3, 57.17],
              [0, 0, 0.5, 0.5], [0.5, 0.5], [0.5, 0.5])

    r = em_double_cluster(x, *theta0)
    param = {'tau1': r[0], 'tau2': r[1],
             'tau3': 1 - r[0] - r[1],
             'muv': r[2], 'mu1': r[3],
             'mu2': r[4], 'sigma02': r[5],
             'sigmax2': r[6], 'sigmav2': r[7]}
    T1, T2, T3 = T(x, *r)

    '''Время джейсона'''

    dictionary = {
        "size_ratio": np.round(param['tau1']/param['tau2'], decimals=2),
        "motion": {"ra": np.round(param['muv'][0], decimals=2),
                   "dec": np.round(param['muv'][1], decimals=2)},
        "clusters": [
            {
                "center": {"ra": np.round(param['mu1'][0], decimals=2),
                           "dec": np.round(param['mu1'][1], decimals=2)},
                },
            {
                "center": {"ra": np.round(param['mu2'][0], decimals=2),
                           "dec": np.round(param['mu2'][1], decimals=2)},
                }
            ]
        }

    with open('per.json', 'w') as f:
        json.dump(dictionary, f, indent=2)

    '''Время графиков'''

    fig, (velocities, loc) = plt.subplots(1, 2, figsize=(14, 5))

    dotsv = velocities.scatter(v1, v2, c=T1+T2, cmap="Oranges")
    add_colorbar(dotsv, 'top')
    circlev = matplotlib.patches.Circle(param['muv'],
                                        np.sqrt(param['sigmav2'][0]),
                                        color='orange', fill=False)
    velocities.add_patch(circlev)
    velocities.xaxis.set_minor_locator(AutoMinorLocator(4))
    velocities.yaxis.set_minor_locator(AutoMinorLocator(4))
    velocities.grid(which='minor', linestyle="--")
    velocities.grid(which='major', linestyle=':')
    velocities.set_xlabel('mu_right ascension', fontproperties=style_default)
    velocities.set_ylabel('mu_declination', fontproperties=style_default)
    velocities.set_title('Velocities distribution', fontproperties=style_BIG)
    dotsx = loc.scatter(x1, x2, c=T1 + T2, cmap="Blues")
    loc.plot(r[3][0], r[3][1], 'o', ms=4, color='blue')
    loc.plot(r[4][0], r[4][1], 'o', ms=4, color='blue')
    loc.set_xlabel('right ascension', fontproperties=style_default)
    loc.set_ylabel('declination', fontproperties=style_default)
    loc.set_title('Coordinates distribution', fontproperties=style_BIG)
    circleh = matplotlib.patches.Circle(param['mu1'],
                                        np.sqrt(param['sigmax2'][0]),
                                        color='blue', fill=False)
    circlex = matplotlib.patches.Circle(param['mu2'],
                                        np.sqrt(param['sigmax2'][0]),
                                        color='blue', fill=False)
    loc.add_patch(circleh)
    loc.add_patch(circlex)
    loc.xaxis.set_minor_locator(AutoMinorLocator(4))
    loc.yaxis.set_minor_locator(AutoMinorLocator(4))
    loc.grid(which='minor', linestyle="--")
    loc.grid(which='major', linestyle=':')
    add_colorbar(dotsx, 'top')
    fig.savefig('per.png')
