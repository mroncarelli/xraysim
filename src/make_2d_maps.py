from matplotlib.colors import LogNorm

from proj2d.makemap import makemap
import matplotlib.pyplot as plt

indir = '/Users/mauro/XRISM/TheThreeHundred/Gadget3PESPH/NewMDCLUSTER_0322/'
infile = 'snap_128'

#res = makemap(indir + infile, 'rho', struct = True, npix=512, center=[500000., 500000.], size=10000., tcut=-1.e5, sample=100)
res = makemap(indir + infile, 'rho', struct = True, npix=512)
print(res['map'].min(), res['map'].max())
#plt.imshow(res, norm=LogNorm(vmin=1.e6, vmax=1.e8))
#plt.imshow(res['map'], norm=LogNorm(vmin=1.e6, vmax=1.e8), extent=[res['xrange'][0], res['xrange'][1], res['yrange'][0], res['yrange'][1]])
plt.imshow(res['map'].transpose(), origin='lower', extent=[res['xrange'][0], res['xrange'][1], res['yrange'][0], res['yrange'][1]])
plt.show()
print('pippo')
