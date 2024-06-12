# INPUT FILES FOR TEST

This document explains the use of the files in this directory, that serve as an input for the tests in the parent 
directory. They are all meant to be both light and representative of the ones used for scientific results.

### `bapec_fakeit_for_test.pha`  
This file was created with `fakeit` using the `bapec` model (*kT* = 5 keV, *Abund* = 0.3 Z<sub>Sun</sub>, *z* = 0.2, 
*b* = 300 km/s, *norm* = 0.1). The response files are `LAD_M4_v2.0.rmf` and `LAD_M4_v2.0.arf`, the exposure was set to 
1500 s.  

### `LAD_M4_v2.0.arf`, `LAD_M4_v2.0.rmf`  
*LOFT-LAD* response files. They was chosen between the ones available in Xspec as they are the largest ones below 10MB
(the Github file size limit).  

### `snap_Gadget_sample`, `snap_sample.hdf5`
A *Gadget* snapshot containing a galaxy cluster simulated at low resolution in Gadget-2 and HDF5 format, 
respectively.  

### `test_emission_table.fits`
An emission table with `APEC` spectra at *Abund* = 0.3 Z<sub>Sun</sub>. The table contains spectra at 6 different 
redshift and 40 different temperatures.
