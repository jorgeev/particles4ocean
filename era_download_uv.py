import cdsapi

c = cdsapi.Client()
for yy in years:
    for mm in months:
        c.retrieve(
            'reanalysis-era5-single-levels',
			{
				'product_type': 'reanalysis',
				'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind'],
				'year': '2023',
				'month': '08',
                'day' : ['12', '13', '14', '15'],
				'time':  ['00:00', '01:00', '02:00',
                          '03:00', '04:00', '05:00',
                          '06:00', '07:00', '08:00',
                          '09:00', '10:00', '11:00',
                          '12:00', '13:00', '14:00',
                          '15:00', '16:00', '17:00',
                          '18:00', '19:00', '20:00',
                          '21:00', '22:00', '23:00',],
				'format': 'netcdf',
                'area': [ 32, -98, 18, -77,],
            },
			'era5_uv10.nc')
            
