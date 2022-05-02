# SZ_TNG
Old scripts for creating SZ maps with IllustrisTNG/MillenniumTNG

# Procedure
1. `save_field.py`: Save the fields we need for the calculation. Note that we assume gas cell temperature and velocity is the electron temperature and velocity. Also, we don't assume that the cells are spherical, but rather save their differential volume dV, which is obtained as Masses/Density of PartType0 (i.e. the gas cells).
Fields needed for each gas cell: electron temperature, electron number density, electron peculiar velocity, volume, coordinates
2. `get_sz_maps.py`: Compute the contribution of each gas cell to the y (tSZ) and b (kSZ) signals. Units are a bit mixed, but they cancel out. Divide by the angular distance (d_A) squared to get values as observed. Flatten in three directions (x, y and z) using a histograming method onto a two-dimensional map of size 10,000^2.
3. `apply_beam.py`: Apply Gaussian beam of 1.3 or 2.1 arcmin to the maps and save beamed images and maps.

# Questions
- Is my approach to handling the volume contribution of each cell correct?
- Do the units check out at the end (according to my tests, yes)?
- Is the division by the angular distance squared necessary? 
- The fact that we are dealing with comoving units shouldn't make a difference until we apply the beam (it gets canceled).

# TNG vs MTNG
There is a difference in both the Unit Time and Unit Length units by 1000. However that doesn't affect any of the SZ calculations, as these get canceled out for the tSZ and kSZ calculations. Note that unit_c should actually just be 10^10 because it is in units of (km/s)^2 already and (km/s)^2 -> 10^10 (cm/s)^2. In that sense, there is a mistake in the TNG FAQ: the units are not kpc/Gyr to km/s, 1.024, (Unit_Length/Unit_Time)^2*Unit_Mass, but rather km/s to cm/s.