from astropy.io import fits
from astropy import cosmology
from src.pkg.sphprojection.mapping import make_speccube
from src.pkg.specutils.sixte import cube2simputfile
import os
from src.pkg.gadgetutils.phys_const import keV2K

indir = os.environ.get('HOME') + '/XRISM/TheThreeHundred/Gadget3PESPH/NewMDCLUSTER_0322/'
outdir = os.environ.get('HOME') + '/XRISM/Simput/'
fileList = ['snap_128']
#spfile = os.environ.get('XRAYSIM') + '/tests/data/test_emission_table.fits'
spfile = os.environ.get('HOME') + '/XRISM/Emission_Tables/xrism_resolve_emission_table_z0.03-0.14.fits'
center_3d = [500158., 500156, 500250]
#size = 0.9780893857  # [deg] (5 h^-1 Mpc per side at z=0.1, with Omega_m=0.3)
size = 0.1  # [deg] (6 arcmin, 2x XRISM Resolve FOV)
redshift = 0.1
nsample = 1
tcut = 1.e6  # [K]
t_iso_keV = 5.7322307  # [keV]
cosmo = cosmology.FlatLambdaCDM(H0=100., Om0=0.3)
d_c = cosmo.comoving_distance(redshift).to_value()  # [h^-1 Mpc]

# X-IFU parameters
npix = 240

# XRISM-Resolve
#npix = 30

for file in fileList:

    infile = indir + file
    for proj in ['x']:
        outfile = outdir + 'spcube_' + file + '_' + str(npix) + '_' + proj + '_iso' + str(t_iso_keV) + '.simput'
        hduList = fits.HDUList()

        if proj == 'x':
            center = [center_3d[1], center_3d[2]]
        elif proj == 'y':
            center = [center_3d[2], center_3d[0]]
        else:
            center = [center_3d[0], center_3d[1]]

        spcubeStruct = make_speccube(infile, spfile, size, npix, redshift=redshift, center=center, proj=proj, tcut=tcut,
                                     nsample=nsample, progress=True, nh=0.0, isothermal=t_iso_keV * keV2K, novel=False)
        print(spcubeStruct.get('data').min(), spcubeStruct.get('data').max())
        cube2simputfile(spcubeStruct, outfile)
