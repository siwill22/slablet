"""Drive slablet with a GPlates reconstruction (needs pygplates).

    models          which reconstruction: gprm, plate_model_manager or explicit files (config block)
    trench_samples  subduction-zone samples per Myr, cached
    seafloor_tracker  seafloor ages at the trenches and plate velocities, cached
    lithosphere     lithosphere thickness from those ages and the continent / craton polygons
    run_mantle      the driver: config -> mantle_temp_<age>Ma.nc
"""
