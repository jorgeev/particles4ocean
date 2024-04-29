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

from njordr.models import njordr_water

model = njordr_water(lon0=-91.88, lat0=19, 
                     particles=100000, dt=3600, 
                     difussivity=0.1, duration=24)
print(model.difussivity)

model.run()

fig = plt.figure(dpi=150)
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

for ii in range(model.total_outputs):
    ax.set_extent([-92.8,-90.7,18.5,19.2])
    ax.scatter(model.lon_out[ii], model.lat_out[ii], s=0.5, c='k')
    ax.coastlines()
    plt.pause(0.2)
    if ii < model.total_outputs-1:
        plt.cla()

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
# ax.set_extent([-99.5214,-78.5589,16.5369,32.4075])
# gl = ax.gridlines(crs=ccrs.PlateCarree(), 
#                   draw_labels=True, 
#                   linewidth=0.6, alpha=0.8, linestyle='--')
# ax.scatter(model.lon.get(), model.lat.get(), s=0.5, c='k')
# ax.coastlines()
# fig.show()
