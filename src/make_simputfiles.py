from astropy.io import fits
from astropy import cosmology
from src.pkg.sphprojection.mapping import make_speccube, write_speccube
from src.pkg.specutils.sixte import cube2simputfile
import os

indir = os.environ.get('HOME') + '/XRISM/TheThreeHundred/Gadget3PESPH/NewMDCLUSTER_0322/'
cubedir = os.environ.get('HOME') + '/XRISM/Spec_Cubes/'
outdir = os.environ.get('HOME') + '/XRISM/Simput/'
fileList = ['snap_128']
#spfile = os.environ.get('XRAYSIM') + '/tests/data/test_emission_table.fits'
spfile = os.environ.get('HOME') + '/XRISM/Emission_Tables/xrism_resolve_emission_table_z0.03-0.14.fits'
npix = 30
center_3d = [500158., 500156, 500250]
#size = 0.9780893857  # [deg] (5 h^-1 Mpc per side at z=0.1, with Omega_m=0.3)
size = 0.1  # [deg] (6 arcmin, 2x XRISM Resolve FOV)
redshift = 0.1
nsample = 1
tcut = 1.e6
cosmo = cosmology.FlatLambdaCDM(H0=100., Om0=0.3)
d_c = cosmo.comoving_distance(redshift).to_value()  # [h^-1 Mpc]

for file in fileList:

    infile = indir + file
    for proj in ['x', 'y', 'z']:
        filename = 'spcube_' + file + '_' + str(npix) + '_' + proj
        cubefile = cubedir + filename + '.fits'
        outfile = outdir + filename + '.simput'
        hduList = fits.HDUList()

        if proj == 'x':
            center = [center_3d[1], center_3d[2]]
        elif proj == 'y':
            center = [center_3d[2], center_3d[0]]
        else:
            center = [center_3d[0], center_3d[1]]

        spcubeStruct = make_speccube(infile, spfile, size, npix, redshift=redshift, center=center, proj=proj, tcut=tcut,
                                     nsample=nsample, progress=True, nh=0.02)
        print(spcubeStruct.get('data').min(), spcubeStruct.get('data').max())
        write_speccube(spcubeStruct, cubefile)
        cube2simputfile(spcubeStruct, outfile)
