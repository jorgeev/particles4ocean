# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 20:38:53 2024

@author: jevza
"""

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

file = 'generic_simulation.nc'

ds = xr.open_dataset(file)

for ii, tt in enumerate(ds.obs.data):
    fig = plt.figure(dpi=100)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    aux = ds.sel(obs=ii)
    ax.set_extent([-87.5,-86.,19.5,21.5])
    gl = ax.gridlines(crs=ccrs.PlateCarree(), 
                      draw_labels=True, 
                      linewidth=0.6, alpha=0.8, linestyle='--')
    ax.scatter(aux.lon.data, aux.lat.data, s=0.5, c='k')
    ax.scatter(-87, 20, s=5, marker='x', c='r')
    ax.coastlines()
    plt.savefig(F'out\eg_{ii:03d}.jpg',  bbox_inches='tight', pad_inches=0.1, pil_kwargs={'quality':95})
    ax.cla()
    fig.clf()
    plt.close()
#fig.show()
