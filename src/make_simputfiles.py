from astropy.io import fits
from astropy import cosmology
from src.pkg.sphprojection.mapping import make_speccube, cube2simputfile
import os

indir = '/Users/mauro/XRISM/TheThreeHundred/Gadget3PESPH/NewMDCLUSTER_0322/'
outdir = '/Users/mauro/XRISM/Maps'
fileList = ['snap_128']
spfile = os.environ.get('XRAYSIM') + '/tests/data/test_emission_table.fits'
npix = 5
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
        outfile = outdir + '/spcube_' + file + '_' + str(npix) + '_' + proj + '.simput'
        hduList = fits.HDUList()

        spcubeStruct = make_speccube(infile, spfile, size, npix, redshift=redshift, center=center, proj=proj, tcut=tcut,
                                     nsample=nsample, struct=True, progress=True)
        print(spcubeStruct.get('data').min(), spcubeStruct.get('data').max())
        cube2simputfile(spcubeStruct, outfile)
