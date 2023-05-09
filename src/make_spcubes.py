from astropy.io import fits
from astropy import cosmology
from src.pkg.sphprojection.mapping import make_speccube
import os

indir = '/Users/mauro/XRISM/TheThreeHundred/Gadget3PESPH/NewMDCLUSTER_0322/'
outdir = '/Users/mauro/XRISM/Maps'
fileList = ['snap_128']
spfile = os.environ.get('XRAYSIM') + '/tests/data/emission_table.fits'
npix = 128
center = [500e3, 500e3]
size = 0.9780893857  # [deg] (5 h^-1 Mpc per side at z=0.1, with Omega_m=0.3)
redshift = 0.1
nsample = 1
tcut = 1.e6
cosmo = cosmology.FlatLambdaCDM(H0=100., Om0=0.3)
d_c = cosmo.comoving_distance(redshift).to_value()  # [h^-1 Mpc]

for file in fileList:

    infile = indir + file
    for proj in ['x', 'y', 'z']:
        outfile = outdir + '/spcube_' + file + '_' + str(npix) + '_' + proj + '.fits'
        hduList = fits.HDUList()

        spcubeStruct = make_speccube(infile, spfile, size, npix, redshift=redshift, center=center, proj=proj, tcut=tcut,
                                     nsample=nsample, struct=True, progress=True)
        print(spcubeStruct.get('data').min(), spcubeStruct.get('data').max())
        energy = spcubeStruct.get('energy')  # [keV]
        d_ene = spcubeStruct.get('energy_interval')  # [keV]
        npix = spcubeStruct.get('data').shape[0]
        nene = spcubeStruct.get('data').shape[2]

        # Output

        # Primary
        hduList.append(fits.PrimaryHDU(spcubeStruct.get('data')))
        hduList[-1].header.set('INFO', 'Created with Python astropy')
        hduList[-1].header.set('SIM_FILE', infile)
        hduList[-1].header.set('SP_FILE', spfile)
        hduList[-1].header.set('PROJ', proj)
        hduList[-1].header.set('Z_COS', redshift)
        hduList[-1].header.set('D_C', d_c, '[h^-1 Mpc]')
        hduList[-1].header.set('NPIX', npix)
        hduList[-1].header.set('NENE', nene)
        hduList[-1].header.set('ANG_PIX', spcubeStruct.get('pixel_size'), '[arcmin]')
        hduList[-1].header.set('ANG_MAP', spcubeStruct.get('size'), '[deg]')
        hduList[-1].header.set('E_MIN', energy[0] - d_ene[0], '[' + spcubeStruct.get('energy_units') + ']')
        hduList[-1].header.set('E_MAX', energy[-1] + d_ene[-1], '[' + spcubeStruct.get('energy_units') + ']')
        hduList[-1].header.set('FLAG_ENE', 1 if spcubeStruct.get('flag_ene') else 0)
        hduList[-1].header.set('UNITS', '[' + spcubeStruct.get('units') + ']')

        # Extension 1
        hduList.append(fits.ImageHDU(spcubeStruct.get('energy'), name='Energy'))
        hduList[-1].header.set('NENE', nene)
        hduList[-1].header.set('UNITS', '[' + spcubeStruct.get('energy_units') + ']')

        # Extension 2
        hduList.append(fits.ImageHDU(spcubeStruct.get('energy_interval'), name='En. interval'))
        hduList[-1].header.set('NENE', nene)
        hduList[-1].header.set('UNITS', '[' + spcubeStruct.get('energy_units') + ']')

        # Writing FITS file
        hduList.writeto(outfile, overwrite=True)
