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

model = njordr_waterwind(lon0=-87, lat0=20, 
                         particles=10000, dt=60, outputstep=3600,
                         start_time='2023-08-12 00:00:00',
                         spill_duration=10,
                         difussivity=0.1, duration=40)
print(model.difussivity)
model.run()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent([-99.5214,-78.5589,16.5369,32.4075])
gl = ax.gridlines(crs=ccrs.PlateCarree(), 
                  draw_labels=True, 
                  linewidth=0.6, alpha=0.8, linestyle='--')
ax.scatter(model.lon.get(), model.lat.get(), s=0.5, c='k')
ax.coastlines()
ax.set_aspect(1)
fig.show()
