import numpy as np
from astropy.io import fits
from sphprojection.mapping import makemap

indir = '/Users/mauro/XRISM/TheThreeHundred/Gadget3PESPH/NewMDCLUSTER_0322/'
outdir = '/Users/mauro/XRISM/Maps'
fileList = ['snap_128']
npix = 16
center = [500e3, 500e3]
size = 5.e3
nsample = None
tcut = 1.e6

for file in fileList:

    infile = indir + file
    for proj in ['x', 'y', 'z']:
        outfile = outdir + '/' + file + '_' + str(npix) + '_' + proj + '.fits'

        hduList = fits.HDUList()

        mapStruct = makemap(infile, 'Tmw', struct=True, npix=npix, center=center, size=size, nsample=nsample, tcut=tcut, progress=True)
        xCoord = np.linspace(mapStruct['xrange'][0], mapStruct['xrange'][1], npix,
                             endpoint=False) + 0.5 * mapStruct['pixel_size']
        yCoord = np.linspace(mapStruct['yrange'][0], mapStruct['yrange'][1], npix,
                             endpoint=False) + 0.5 * mapStruct['pixel_size']
        coordMap = np.array([xCoord, yCoord])

        # Primary: contains only the coordinates
        hduList.append(fits.PrimaryHDU(coordMap))
        hduList[-1].header.set('SIM_FILE', infile)
        if nsample:
            hduList[-1].header.set('NSAMPLE', nsample)
        if tcut:
            hduList[-1].header.set('T_CUT', tcut)
        hduList[-1].header.set('PROJ', proj)
        hduList[-1].header.set('NX', npix)
        hduList[-1].header.set('NY', npix)
        hduList[-1].header.set('L_PIX', mapStruct['pixel_size'])
        hduList[-1].header.set('UNITS', '[' + mapStruct['coord_units'] + ']')

        # Extension 1
        hduList.append(fits.ImageHDU(mapStruct['norm'], name='Surface density'))
        hduList[-1].header.set('EXPR', 'Int(rho*dl)')
        hduList[-1].header.set('UNITS', '[' + mapStruct['norm_units'] + ']')

        # Extension 3 (provisionally in position 2, will go to 3 after the emission measure is inserted
        hduList.append(fits.ImageHDU(mapStruct['map'], name='Mass-weighted temperature'))
        hduList[-1].header.set('EXPR', 'Int(rho*T*dl)/Int(rho*dl)')
        hduList[-1].header.set('UNITS', '[' + mapStruct['units'] + ']')

        mapStruct = makemap(infile, 'Tew', struct=True, npix=npix, center=center, size=size, nsample=nsample, tcut=tcut, progress=True)

        # Extension 2
        hduList.insert(2, fits.ImageHDU(mapStruct['norm'], name='Emission measure'))
        hduList[2].header.set('EXPR', 'Int(rho^2*dl)')
        hduList[2].header.set('UNITS', '[' + mapStruct['norm_units'] + ']')

        # Extension 4
        hduList.append(fits.ImageHDU(mapStruct['map'], name='Emission-weighted temperature'))
        hduList[-1].header.set('EXPR', 'Int(rho^2*T*dl)/Int(rho^2*dl)')
        hduList[-1].header.set('UNITS', '[' + mapStruct['units'] + ']')

        mapStruct = makemap(infile, 'Tsl', struct=True, npix=npix, center=center, size=size, nsample=nsample, tcut=tcut, progress=True)

        # Extension 5
        hduList.append(fits.ImageHDU(mapStruct['map'], name='Spectroscopic-like temperature'))
        hduList[-1].header.set('EXPR', 'Int(rho^2*T^0.25*dl)/Int(rho^2*T^-0.75*dl)')
        hduList[-1].header.set('UNITS', '[' + mapStruct['units'] + ']')

        # Extension 6
        hduList.append(fits.ImageHDU(mapStruct['norm'], name='Spectroscopic-like weight'))
        hduList[-1].header.set('EXPR', 'Int(rho^2*T^-0.75*dl)')
        hduList[-1].header.set('UNITS', '[' + mapStruct['norm_units'] + ']')

        mapStruct = makemap(infile, 'wmw', struct=True, npix=npix, center=center, size=size, nsample=nsample, tcut=tcut, progress=True)

        # Extension 7
        hduList.append(fits.ImageHDU(mapStruct['map2'], name='Mass-weighted velocity'))
        hduList[-1].header.set('EXPR', 'Int(rho*v*dl)/Int(rho*dl)')
        hduList[-1].header.set('UNITS', '[' + mapStruct['map2_units'] + ']')

        # Extension 9 (provisionally in position 8, will go to 9 after the emission weighted velocity is inserted
        hduList.append(fits.ImageHDU(mapStruct['map'], name='Mass-weighted velocity dispersion'))
        hduList[-1].header.set('EXPR', 'SQRT( Int(rho*(v-v0)^2*dl)/Int(rho*dl) )')
        hduList[-1].header.set('UNITS', '[' + mapStruct['units'] + ']')

        mapStruct = makemap(infile, 'wew', struct=True, npix=npix, center=center, size=size, nsample=nsample, tcut=tcut, progress=True)

        # Extension 8
        hduList.insert(8, fits.ImageHDU(mapStruct['map2'], name='Emission-weighted velocity'))
        hduList[8].header.set('EXPR', 'Int(rho^2*v*dl)/Int(rho^2*dl)')
        hduList[8].header.set('UNITS', '[' + mapStruct['map2_units'] + ']')

        # Extension 10
        hduList.append(fits.ImageHDU(mapStruct['map'], name='Emission-weighted velocity dispersion'))
        hduList[-1].header.set('EXPR', 'SQRT( Int(rho^2*(v-v0)^2*dl)/Int(rho^2*dl) )')
        hduList[-1].header.set('UNITS', '[' + mapStruct['units'] + ']')

        # Writing FITS file
        hduList.writeto(outfile, overwrite=True)
