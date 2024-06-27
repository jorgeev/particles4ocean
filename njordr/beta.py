# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 19:23:58 2024

@author: jevza
"""

import numpy as np
import xarray as xr
import scipy as cp
from scipy.interpolate import interpn
from netCDF4 import Dataset
from datetime import timedelta, datetime
from cftime import date2num, num2date

class wind_model():
    def __init__(self,
                 particles:int=12000, dt:int=1200, 
                 lat0:float=24, lon0:float=-88.5,
                 radius:float=100, difussivity:float=0.1,
                 start_time:str='2023-08-12T00:00:00', duration:int=36,
                 spill_duration:float=1,
                 outputstep:int=3600,
                 earth_radius:float=6371000.,
                 name:str='wind_simulation',
                 wind:str='wrf_file.nc'
                 ):
        """
        particules     : Total number of particules the model will use
        dt             : Size of the time step, in seconds
        lat0           : Source latitude for the model
        lon0           : Source longitude for the model
        radius         : Initial disperison radius when initializing new particules
        start_time     : Start time for the simulation
        duration       : Duration of the simulation in hours "duration<=particles"
        spill_duration : Duration of the spill in hours
        outputstep     : Steps where the outputs will be saved "outputstep%dt==0"
        earth_radius   : Earth radius used to correct x
        name           : Name of the simulation
        windv          : Path to netCDF
        Currently new particles are added every timestep
        """
        self.particles = particles
        self.trajectories = np.arange(particles)
        self.lat0 = lat0
        self.lon0 = lon0
        self.lon = np.full(particles, np.nan)
        self.lat = np.full(particles, np.nan)
        self.dt = dt
        self.earth_radius = earth_radius
        self.earth_radius_rad = 180 / (self.earth_radius * np.pi)
        self.radius = radius
        self.difussivity = difussivity / np.sqrt(dt)
        self.duration = int(duration * 3600)
        self.spill_duration = int(spill_duration * 3600)
        self.start_time = start_time
        self.aux_starttime = datetime.strptime(self.start_time, '%Y-%m-%d %H:%M:%S')
        self.outputstep = outputstep
        self.wind = wind
        self.case_name = name
        
        # How many timestep are needed to finish the simulation
        self.total_steps = int(self.duration / self.dt)
        
        # How many outputs will we get
        self.total_outputs = int(self.duration / self.outputstep)
        
        # How many times will we add particles to the simulation
        self.spill_steps = int(self.spill_duration / self.dt)
        
        # Index to initialize new particles
        self.part_idx = np.array_split(self.trajectories, self.spill_steps + 1)
        self.current_idx = self.part_idx[0]
        self.len_cidx = self.current_idx.shape[0]
        self.part_next_id = 1
        
        # Initialize first particles
        self.seedparticles(self.current_idx)
        self.wind, self.current_time = self.preprocess_nc(wind)
        
        # To save in a netcdf4 file
        self.create_netcdf()
        self.nclat[:, 0] = self.lat
        self.nclon[:, 0] = self.lon
        self.nctime[0] = date2num(self.aux_starttime + timedelta(seconds=self.current_time), units=self.nctime.units)
        self.save_idx = 1
        
    def preprocess_nc(self, wind):
        ds = xr.open_dataset(wind)
        ds_start_time = ds['XTIME'].data[0]
        new_times = np.float32((ds['XTIME'].data - ds['XTIME'].data[0])) / 1000000000 
        start_time = np.float32(np.datetime64(self.start_time) - ds_start_time) /  1000000000
        wind = ds.assign_coords(XTIME=new_times)
        
        lon = ds.XLONG.data[0,0]
        lat = ds.XLAT.data[0,:,0]
        
        wind2 = xr.Dataset(coords={
            'lon':(('lon'), lon),
            'lat':(('lat'), lat),
            'time': (('time'), new_times),})
        
        wind2['U10'] = (('time', 'lat', 'lon'), wind['U10'].data)
        wind2['V10'] = (('time', 'lat', 'lon'), wind['V10'].data)
        
        return wind2, start_time
        
    def mt2deg(self, distance:float, lat:float, axis:str):
        if axis == 'x':
            degs = self.earth_radius_rad * distance * np.cos(np.deg2rad(lat))
        else:
            degs = self.earth_radius_rad * distance
        return degs
    
    def seedparticles(self, target_idx):
        self.lon[target_idx] = self.lon0 + self.mt2deg(self.radius, self.lat0, axis='x') * np.random.normal(size=target_idx.shape[0])
        self.lat[target_idx] = self.lat0 + self.mt2deg(self.radius, self.lat0, axis='y') * np.random.normal(size=target_idx.shape[0])
    
    def interp_uvt(self, time):
        # Reduce the data to a smaller interpolation
        time0 = time - 3700
        time1 = time + 3700
        Xp = self.lon[self.current_idx]
        Yp = self.lat[self.current_idx]
        X_max = np.max(Xp)+0.1
        X_min = np.min(Xp)-0.1
        Y_max = np.max(Yp)+0.1
        Y_min = np.min(Yp)-0.1
        target_time = np.zeros(self.len_cidx) + time
        # TODO Fix interpolation
        windcp = self.wind.sel(time=slice(time0, time1))
        
        # Create cupy vars
        LLat = np.array(windcp.lat.data)
        LLon = np.array(windcp.lon.data)
        TTime = np.array(windcp.time.data)
        U = np.array(windcp['U10'].data)
        V = np.array(windcp['V10'].data)
        nc_coords = (TTime, LLat, LLon)

        # interpolate fields
        target_u = interpn(nc_coords, U, (target_time, Yp, Xp))
        target_v = interpn(nc_coords, V, (target_time, Yp, Xp))
        return target_u, target_v
    
    def loc_nan(self, lon, lat):
        self.current_idx = self.current_idx[np.isnan(self.lat[self.current_idx]) != True]
        self.len_cidx = self.current_idx.shape[0]
    
    def step(self):
        old_lon = self.lon[self.current_idx].copy()
        old_lat = self.lat[self.current_idx].copy()
        uu, vv = self.interp_uvt(self.current_time)
        self.lon[self.current_idx] += self.mt2deg(uu*self.dt, self.lat[self.current_idx], 'x') + uu * self.difussivity * np.random.uniform(-1, 1, size=self.len_cidx)
        self.lat[self.current_idx] += self.mt2deg(vv*self.dt, self.lat[self.current_idx], 'y') + vv * self.difussivity * np.random.uniform(-1, 1, size=self.len_cidx)
        self.current_time += self.dt    

        # Initialize next batch of particles
        if self.part_next_id < self.spill_steps + 1:
            self.seedparticles(self.part_idx[self.part_next_id])
            # Update main index of particles
            self.current_idx = np.concatenate((self.current_idx, self.part_idx[self.part_next_id]))
            self.part_next_id += 1 # Prepare next batch
        
        self.loc_nan(old_lon, old_lat)    
        
    def create_netcdf(self): 
        vault_file = F'{self.case_name}.nc'
        vault = Dataset(vault_file, 'w', format='NETCDF4')
        vault.createDimension("obs", None)
        vault.createDimension("traj", self.particles)
        traj = vault.createVariable("trajectory", "u8", ("traj"))
        self.nclat = vault.createVariable("lat", "f8", ("traj","obs"))
        self.nclon = vault.createVariable("lon", "f8", ("traj","obs"))
        self.nctime = vault.createVariable("time", "f8", ("obs"))
        self.nctime.units = F"seconds since {self.start_time}"
        self.nctime.standard_name = 'time'
        traj[:] = self.trajectories
        traj.cf_role = "trajectory_id"
        traj.units = "1"
        self.nclat.units = 'degrees_north'
        self.nclat.standard_name = 'latitude'
        self.nclat.long_name = 'latitude'
        self.nclon.units = 'degrees_east'
        self.nclon.standard_name = 'longitude'
        self.nclon.long_name = 'longitude'
        vault.Conventions = "CF-1.6"
        vault.standard_name_vocabulary = "CF-1.6"
        vault.featureType = "trajectory"
        vault.history = F"Created {np.datetime64('now')}"
        vault.source = "Output from simulation with njordr"
        vault.model_url = "https://github.com/jorgeev/particules4ocean"
        vault.time_coverage_start = "2023-08-12T00:00:00"
        vault.time_step_calculation = F"{str(timedelta(seconds=self.dt))}"
        vault.time_step_output = F"{str(timedelta(seconds=self.outputstep))}"
        self.vault = vault
        
    def run(self):
        for ii in range(self.total_steps):
            print(ii, self.current_time + self.dt)
            self.step()
        
            if self.current_time%self.outputstep==0:
                self.nclat[:, self.save_idx] = self.lat
                self.nclon[:, self.save_idx] = self.lon
                self.nctime[self.save_idx] = date2num(self.aux_starttime + timedelta(seconds=self.current_time), units=self.nctime.units)
                self.save_idx += 1
        self.vault.close()
    