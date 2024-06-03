# -*- coding: utf-8 -*-
"""
Created on Sun May 26 21:09:25 2024

@author: jevza
"""

import cupy as cp
import xarray as xr
import cupy_xarray
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from njordr.models import njordr_waterwind

model = njordr_waterwind(lon0=-87., lat0=20, 
                         particles=100000, dt=300, outputstep=3600,
                         start_time='2023-08-12 00:00:00',
                         spill_duration=40, windage=0.02,
                         difussivity=0.05, duration=40)
print(model.difussivity)
model.run()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent([-87.5,-86.,19.5,21.5])
gl = ax.gridlines(crs=ccrs.PlateCarree(), 
                  draw_labels=True, 
                  linewidth=0.6, alpha=0.8, linestyle='--')
ax.scatter(model.lon.get(), model.lat.get(), s=0.5, c='k')
ax.scatter(-87, 20, s=5, marker='x', c='r')
ax.coastlines()
ax.set_aspect(1)
#plt.savefig('eg_01.jpg',  bbox_inches='tight', pad_inches=0.1, pil_kwargs={'quality':95})
fig.show()
