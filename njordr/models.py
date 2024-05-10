# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 15:34:46 2024

@author: je.velasco
"""
import cupy as cp
from cupyx.scipy.interpolate import interpn
import numpy as np
import xarray as xr
import cupy_xarray
from netCDF4 import Dataset
from datetime import timedelta, datetime
from cftime import date2num, num2date

class njordr_water():
    def __init__(self,
                 particles:int=12000, dt:int=1200, 
                 lat0:float=24, lon0:float=-88.5,
                 radius:float=100, difussivity:float=0.1,
                 start_time:str='2023-08-12T00:00:00', duration:int=36,
                 spill_duration:float=1,
                 outputstep:int=3600,
                 earth_radius:float=6371000.,
                 name:str='generic_simulation',
                 water:str='hycom_hd.nc'
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
        water          : Path to netCDF
        Currently new particles are added every timestep
        """
        self.particles = particles
        self.trajectories = cp.arange(particles)
        self.lat0 = lat0
        self.lon0 = lon0
        self.lon = cp.zeros(particles)
        self.lat = cp.zeros(particles)
        self.dt = dt
        self.earth_radius = earth_radius
        self.earth_radius_rad = 180 / (self.earth_radius * cp.pi)
        self.radius = radius
        self.difussivity = difussivity / np.sqrt(dt)
        self.duration = int(duration * 3600)
        self.spill_duration = int(spill_duration * 3600)
        self.start_time = start_time
        self.aux_starttime = datetime.strptime(self.start_time, '%Y-%m-%d %H:%M:%S')
        self.outputstep = outputstep
        self.water = water
        self.case_name = name
        
        # How many timestep are needed to finish the simulation
        self.total_steps = int(self.duration / self.dt)
        
        # How many outputs will we get
        self.total_outputs = int(self.duration / self.outputstep)
        
        # How many times will we add particles to the simulation
        self.spill_steps = int(self.spill_duration / self.dt)
        
        # Index to initialize new particles
        self.part_idx = cp.array_split(self.trajectories, self.spill_steps + 1)
        self.current_idx = self.part_idx[0]
        self.len_cidx = self.current_idx.shape[0]
        self.part_next_id = 1
        
        # Initialize first particles
        self.seedparticles(self.current_idx)
        self.water, self.current_time = self.preprocess_water(water)
        
        # To save model outputs and store t0       
        # self.lon_out = np.empty([self.total_outputs + 1, self.particles])
        # self.lat_out = np.empty([self.total_outputs + 1, self.particles])
        # self.lat_out[0] = self.lat.get()
        # self.lon_out[0] = self.lon.get()
        
        # To save in a netcdf4 file
        self.create_netcdf()
        self.nclat[:, 0] = self.lat.get()
        self.nclon[:, 0] = self.lon.get()
        self.nctime[0] = date2num(self.aux_starttime + timedelta(seconds=self.current_time), units=self.nctime.units)
        self.save_idx = 1
    
    def preprocess_water(self, water):
        ds = xr.open_dataset(water)
        ds_start_time = ds.MT.data[0]
        new_times = np.float32((ds.MT.data - ds.MT.data[0])) / 1000000000 
        start_time = np.float32(np.datetime64(self.start_time) - ds_start_time) /  1000000000
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
    
    def loc_nan(self, lon, lat):
        #backup = self.current_idx[cp.isnan(self.lat[self.current_idx]) != False]
        self.current_idx = self.current_idx[cp.isnan(self.lat[self.current_idx]) != True]
        #self.lon[backup] = lon[backup]
        #self.lat[backup] = lat[backup]
        self.len_cidx = self.current_idx.shape[0]
    
    def step(self):
        old_lon = self.lon[self.current_idx].copy()
        old_lat = self.lat[self.current_idx].copy()
        uu, vv = self.interp_uvt(self.current_time)
        self.lon[self.current_idx] += self.mt2deg(uu*self.dt, self.lat[self.current_idx], 'x') + uu * self.difussivity * cp.random.uniform(-1, 1, size=self.len_cidx)
        self.lat[self.current_idx] += self.mt2deg(vv*self.dt, self.lat[self.current_idx], 'y') + vv * self.difussivity * cp.random.uniform(-1, 1, size=self.len_cidx)
        self.current_time += self.dt    

        # Initialize next batch of particles
        if self.part_next_id < self.spill_steps + 1:
            self.seedparticles(self.part_idx[self.part_next_id])
            # Update main index of particles
            self.current_idx = cp.concatenate((self.current_idx, self.part_idx[self.part_next_id]))
            self.part_next_id += 1 # Prepare next batch
        
        self.loc_nan(old_lon, old_lat)    
        
    def create_netcdf(self): 
        vault_file = F'{self.case_name}.nc'
        vault = Dataset(vault_file, 'w', format='NETCDF4')
        # vault.createDimension("time", None)
        vault.createDimension("obs", None)
        vault.createDimension("traj", self.particles)
        traj = vault.createVariable("trajectory", "u8", ("traj"))
        # self.nclat = vault.createVariable("lat", "f8", ("time","trajectory"))
        # self.nclon = vault.createVariable("lon", "f8", ("time","trajectory"))
        # self.nctime = vault.createVariable("time", "f8", ("time"))
        self.nclat = vault.createVariable("lat", "f8", ("traj","obs"))
        self.nclon = vault.createVariable("lon", "f8", ("traj","obs"))
        self.nctime = vault.createVariable("time", "f8", ("obs"))
        self.nctime.units = F"seconds since {self.start_time}"
        self.nctime.standard_name = 'time'
        # self.nctime.calendar = "proleptic_gregorian"
        traj[:] = self.trajectories.get()
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
                self.nclat[:, self.save_idx] = self.lat.get()
                self.nclon[:, self.save_idx] = self.lon.get()
                self.nctime[self.save_idx] = date2num(self.aux_starttime + timedelta(seconds=self.current_time), units=self.nctime.units)
                # self.nctime[ii+1] = self.current_time
                # self.lat_out[self.save_idx] = self.lat.get()
                # self.lon_out[self.save_idx] = self.lon.get()
                self.save_idx += 1
        self.vault.close()
