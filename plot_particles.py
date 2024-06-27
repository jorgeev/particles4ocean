# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 20:38:53 2024

@author: jevza
"""

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

nj_file = 'njorder_v_CICOIL.nc'
cicoil_file = 'CICOIL.nc'
njordr = xr.open_dataset(nj_file)
cicoil = xr.open_dataset(cicoil_file)

for ii, tt in enumerate(njordr.obs.data):
    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    aux1 = njordr.isel(obs=ii)
    aux2 = cicoil.isel(time=ii)
    ax.set_extent([-96.,-93.,18.,19.5])
    gl = ax.gridlines(crs=ccrs.PlateCarree(), 
                      draw_labels=True, 
                      linewidth=0.6, alpha=0.8, linestyle='--')
    ax.scatter(aux1.lon.data, aux1.lat.data, s=0.5, c='k', label='njordr')
    ax.scatter(aux2.lon.data, aux2.lat.data, s=0.5, c='r', label='CICOIL')
    ax.scatter(-87, 20, s=5, marker='x', c='orange')
    plt.legend()
    ax.coastlines()
    plt.savefig(F'out/eg_{ii:03d}.jpg',  bbox_inches='tight', 
                pad_inches=0.1, pil_kwargs={'quality':95})
    ax.cla()
    fig.clf()
    plt.close()
#fig.show()
