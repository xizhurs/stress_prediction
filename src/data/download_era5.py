import cdsapi

dataset = "reanalysis-era5-land-monthly-means"
request = {
    "product_type": ["monthly_averaged_reanalysis"],
    "variable": [
        "2m_temperature",
        "volumetric_soil_water_layer_1",
        "evaporation_from_bare_soil",
        "evaporation_from_vegetation_transpiration",
        "potential_evaporation",
        "surface_pressure",
        "total_precipitation",
        "total_evaporation",
        "surface_net_solar_radiation",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "soil_type",
    ],
    "year": [x for x in range(1980, 2025)],
    "month": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"],
    "time": ["00:00"],
    "area": [53.442401, 3.960336, 51.061045, 7.493940],
    "data_format": "netcdf",
    "download_format": "unarchived",
}


client = cdsapi.Client()
client.retrieve(dataset, request).download("data/era5_NL_monthly.nc")
