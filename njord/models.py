# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 15:34:46 2024

@author: je.velasco
TODO: Fix correct array filling I have missed 1 output
"""
import cupy as cp
from cupyx.scipy.interpolate import interpn
import numpy as np
import xarray as xr
import cupy_xarray

class njord_water():
    def __init__(self,
                 particles:int=12000, dt:int=1200, 
                 lat0:float=24, lon0:float=-88.5,
                 radius:float=100, difussivity:float=0.1,
                 start_time:str='2023-08-12T00:00:00', duration:int=36,
                 outputstep:int=3600,
                 earth_radius:float=6371000.,
                 name:str='generic_simulation',
                 water:str='hycom_hd.nc'
                 ):
        """
        particules  : Total number of particules the model will use
        dt          : Size of the time step, in seconds
        lat0        : Source latitude for the model
        lon0        : Source longitude for the model
        radius      : Initial disperison radius when initializing new particules
        start_time  : Start time for the simulation
        duration    : Duration of the simulation in hours "duration<=particles"
        outputstep  : Steps where the outputs will be saved "outputstep%dt==0"
        earth_radius: Earth radius used to correct x
        name        : Name of the simulation
        water       : Path to netCDF
        Currently new particles are added every timestep
        TODO: Idenpendent particle seeding and saving
        """
        self.particles = particles
        self.trajectories = cp.arange(particles)
        self.lat0 = lat0
        self.lon0 = lon0
        self.lon = cp.zeros(particles)
        self.lat = cp.zeros(particles)
        self.dt = dt
        self.half_dt = self.dt / 2
        self.earth_radius = earth_radius
        self.earth_radius_rad = 180 / (self.earth_radius * cp.pi)
        self.radius = radius
        self.difussivity = difussivity / np.sqrt(dt)
        self.duration = int(duration * 3600)
        self.start_time = start_time
        self.outputstep = outputstep
        self.water = water
        # How many timestep are needed to finish the simulation
        self.total_steps = int(self.duration / self.dt)
        # How many outputs will we get
        self.total_outputs = int(self.duration / self.outputstep)
        # Index to initialize new particles
        self.part_idx = cp.array_split(self.trajectories, self.total_steps)
        self.current_idx = self.part_idx[0]
        self.len_cidx = self.current_idx.shape[0]
        self.part_next_id = 1
        # Initialize first particles
        self.seedparticles(self.current_idx)
        self.water, self.current_time = self.preprocess_water(water)
        # To save model outputs
        self.save_idx = 1
        self.lon_out = np.empty([self.total_outputs, self.particles])
        self.lat_out = np.empty([self.total_outputs, self.particles])
        self.lat_out[0] = self.lat.get()
        self.lon_out[0] = self.lon.get()
    
    def preprocess_water(self, water):
        ds = xr.open_dataset(water)
        ds_start_time = ds.MT.data[0]
        new_times = np.float32((ds.MT.data - ds.MT.data[0])) / 1000000000 
        start_time = np.float32(ds_start_time - np.datetime64(self.start_time)) /  1000000000
        water = ds.assign_coords(MT=new_times)
        return water, start_time
        
    def mt2deg(self, distance:float, lat:float, axis:str):
        if axis == 'x':
            degs = self.earth_radius_rad * distance * cp.cos(cp.deg2rad(lat))
        else:
            degs = self.earth_radius_rad * distance
        return degs
    
    def seedparticles(self, target_idx):
        self.lon[target_idx] = self.lon0 + self.mt2deg(self.radius, self.lat0, axis='x') * cp.random.normal(size=target_idx.shape[0])
        self.lat[target_idx] = self.lat0 + self.mt2deg(self.radius, self.lat0, axis='y') * cp.random.normal(size=target_idx.shape[0])
    
    def interp_uvt(self, time):
        # Reduce the data to a smaller interpolation
        time0 = time - 3700
        time1 = time + 3700
        Xp = self.lon[self.current_idx]
        Yp = self.lat[self.current_idx]
        X_max = cp.max(Xp)+0.1
        X_min = cp.min(Xp)-0.1
        Y_max = cp.max(Yp)+0.1
        Y_min = cp.min(Yp)-0.1
        target_time = cp.zeros(self.len_cidx) + time
        
        watercp = self.water.isel(Depth=0).sel(Latitude=slice(Y_min.get(), Y_max.get()), 
                                               Longitude=slice(X_min.get(), X_max.get()), 
                                               MT=slice(time0, time1))
        
        # Create cupy vars
        LLat = cp.array(watercp.Latitude.data)
        LLon = cp.array(watercp.Longitude.data)
        TTime = cp.array(watercp.MT.data)
        U = cp.array(watercp['u'].data)
        V = cp.array(watercp['v'].data)
        nc_coords = (TTime, LLat, LLon)

        # interpolate fields
        target_u = interpn(nc_coords, U, (target_time, Yp, Xp))
        target_v = interpn(nc_coords, V, (target_time, Yp, Xp))
        return target_u, target_v
    
    def step(self):
        #print(self.current_time)
        uu, vv = self.interp_uvt(self.current_time)
        self.lon[self.current_idx] += self.mt2deg(uu*self.dt, self.lat[self.current_idx], 'x') + uu * self.difussivity * cp.random.uniform(-1, 1, size=self.len_cidx)
        self.lat[self.current_idx] += self.mt2deg(vv*self.dt, self.lat[self.current_idx], 'y') + vv * self.difussivity * cp.random.uniform(-1, 1, size=self.len_cidx)
        self.current_time += self.dt    
        #if self.current_time != 0:
        # print('Seeding new particles')
        self.seedparticles(self.part_idx[self.part_next_id])
        self.current_idx = cp.concatenate((self.current_idx, self.part_idx[self.part_next_id]))
        self.len_cidx = self.current_idx.shape[0]
        self.part_next_id += 1
        
        if self.current_time%self.outputstep==0:
            self.lat_out[self.save_idx] = self.lat.get()
            self.lon_out[self.save_idx] = self.lon.get()
            self.save_idx += 1
                
