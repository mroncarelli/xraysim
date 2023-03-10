import numpy as np
from astropy.io import fits

from proj2d.makemap import makemap

indir = '/Users/mauro/XRISM/TheThreeHundred/Gadget3PESPH/NewMDCLUSTER_0322/'
outdir = '/Users/mauro/XRISM/Maps'
fileList = ['snap_128']
npix = 256
center = [500e3, 500e3]
size = 5.e3
sample = 50

for file in fileList:

    infile = indir + file
    for proj in ['x', 'y', 'z']:
        outfile = outdir + '/' + file + '_' + proj + '.fits'

        hduList = fits.HDUList()

        mapStruct = makemap(infile, 'Tmw', struct=True, npix=npix, center=center, size=size, sample=sample)
        xCoord = np.linspace(mapStruct['xrange'][0], mapStruct['xrange'][1], npix,
                             endpoint=False) + 0.5 * mapStruct['pixel_size']
        yCoord = np.linspace(mapStruct['yrange'][0], mapStruct['yrange'][1], npix,
                             endpoint=False) + 0.5 * mapStruct['pixel_size']
        coordMap = np.array([xCoord, yCoord])

        # Primary: contains only the coordinates
        hduList.append(fits.PrimaryHDU(coordMap))
        hduList[-1].header.set('SIM_FILE', infile)
        hduList[-1].header.set('PROJ', proj)
        hduList[-1].header.set('NX', npix)
        hduList[-1].header.set('NY', npix)
        hduList[-1].header.set('L_PIX', mapStruct['pixel_size'])
        hduList[-1].header.set('UNITS', '[' + mapStruct['coord_units'] + ']')

        # Extension 1
        hduList.append(fits.ImageHDU(mapStruct['norm']))
        hduList[-1].header.set('MAPNAME', 'Surface density')
        hduList[-1].header.set('EXPR', 'Int(rho*dl)')
        hduList[-1].header.set('UNITS', '[' + mapStruct['norm_units'] + ']')

        # Extension 3 (provisionally in position 2, will go to 3 after the emission measure is inserted
        hduList.append(fits.ImageHDU(mapStruct['map']))
        hduList[-1].header.set('MAPNAME', 'Mass-weighted temperature')
        hduList[-1].header.set('EXPR', 'Int(rho*T*dl)/Int(rho*dl)')
        hduList[-1].header.set('UNITS', '[' + mapStruct['units'] + ']')

        mapStruct = makemap(infile, 'Tew', struct=True, npix=npix, center=center, size=size, sample=sample)

        # Extension 2
        hduList.insert(2, fits.ImageHDU(mapStruct['norm']))
        hduList[2].header.set('MAPNAME', 'Emission measure')
        hduList[2].header.set('EXPR', 'Int(rho^2*dl)')
        hduList[2].header.set('UNITS', '[' + mapStruct['norm_units'] + ']')

        # Extension 4
        hduList.append(fits.ImageHDU(mapStruct['map']))
        hduList[-1].header.set('MAPNAME', 'Emission-weighted temperature')
        hduList[-1].header.set('EXPR', 'Int(rho^2*T*dl)/Int(rho^2*dl)')
        hduList[-1].header.set('UNITS', '[' + mapStruct['units'] + ']')

        mapStruct = makemap(infile, 'Tsl', struct=True, npix=npix, center=center, size=size, sample=sample)

        # Extension 5
        hduList.append(fits.ImageHDU(mapStruct['map']))
        hduList[-1].header.set('MAPNAME', 'Spectroscopic-like temperature')
        hduList[-1].header.set('EXPR', 'Int(rho^2*T^0.25*dl)/Int(rho^2*T^-0.75*dl)')
        hduList[-1].header.set('UNITS', '[' + mapStruct['units'] + ']')

        # Extension 6
        hduList.append(fits.ImageHDU(mapStruct['norm']))
        hduList[-1].header.set('MAPNAME', 'Spectroscopic-like weight')
        hduList[-1].header.set('EXPR', 'Int(rho^2*T^-0.75*dl)')
        hduList[-1].header.set('UNITS', '[' + mapStruct['norm_units'] + ']')

        mapStruct = makemap(infile, 'wmw', struct=True, npix=npix, center=center, size=size, sample=sample)

        # Extension 7
        hduList.append(fits.ImageHDU(mapStruct['map2']))
        hduList[-1].header.set('MAPNAME', 'Mass-weighted velocity')
        hduList[-1].header.set('EXPR', 'Int(rho*v*dl)/Int(rho*dl)')
        hduList[-1].header.set('UNITS', '[' + mapStruct['map2_units'] + ']')

        # Extension 9 (provisionally in position 8, will go to 9 after the emission weighted velocity is inserted
        hduList.append(fits.ImageHDU(mapStruct['map']))
        hduList[-1].header.set('MAPNAME', 'Mass-weighted velocity dispersion')
        hduList[-1].header.set('EXPR', 'SQRT( Int(rho*(v-v0)^2*dl)/Int(rho*dl) )')
        hduList[-1].header.set('UNITS', '[' + mapStruct['units'] + ']')

        mapStruct = makemap(infile, 'wew', struct=True, npix=npix, center=center, size=size, sample=sample)

        # Extension 8
        hduList.insert(8, fits.ImageHDU(mapStruct['map2']))
        hduList[8].header.set('MAPNAME', 'Emission-weighted velocity')
        hduList[8].header.set('EXPR', 'Int(rho^2*v*dl)/Int(rho^2*dl)')
        hduList[8].header.set('UNITS', '[' + mapStruct['map2_units'] + ']')

        # Extension 10
        hduList.append(fits.ImageHDU(mapStruct['map']))
        hduList[-1].header.set('MAPNAME', 'Emission-weighted velocity dispersion')
        hduList[-1].header.set('EXPR', 'SQRT( Int(rho^2*(v-v0)^2*dl)/Int(rho^2*dl) )')
        hduList[-1].header.set('UNITS', '[' + mapStruct['units'] + ']')

        # Writing FITS file
        hduList.writeto(outfile, overwrite=True)
