# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 16:19:19 2024

@author: je.velasco
simple layout of the model
"""
import cupy as cp
import xarray as xr
import cupy_xarray
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from njord.models import njord_water

model = njord_water(lon0=-91.88, lat0=19.65, 
                    particles=100000, dt=360, 
                    difussivity=0.0, duration=40)
print(model.difussivity)

for ii in range(model.total_steps-1):
    model.step()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent([-99.5214,-78.5589,16.5369,32.4075])
gl = ax.gridlines(crs=ccrs.PlateCarree(), 
                  draw_labels=True, 
                  linewidth=0.6, alpha=0.8, linestyle='--')
ax.scatter(model.lon.get(), model.lat.get(), s=0.5, c='k')
ax.coastlines()
fig.show()
