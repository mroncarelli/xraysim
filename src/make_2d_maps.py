from matplotlib.colors import LogNorm
import pygadgetreader as pygr
from proj2d.makemap import makemap
import matplotlib.pyplot as plt
import time as t

indir = '/Users/mauro/XRISM/TheThreeHundred/Gadget3PESPH/NewMDCLUSTER_0322/'
infile = 'snap_128'

mass_snap = sum(pygr.readsnap(indir+infile, 'mass', 'gas', units=0)) # [10^10 h^-1 M_Sun]

#res = makemap(indir + infile, 'rho', struct = True, npix=512, center=[500000., 500000.], size=10000., tcut=-1.e5, sample=100)
start = t.time()
res = makemap(indir + infile, 'rho', struct = True, npix=512, center=[500000., 500000.], size=10000., sample=30)
end = t.time()
print(res['map'].min(), res['map'].max(), end-start)
mass_map = sum(sum(res['map'])) * res['pixel_size'] ** 2
#plt.imshow(res, norm=LogNorm(vmin=1.e6, vmax=1.e8))
#plt.imshow(res['map'], norm=LogNorm(vmin=1.e6, vmax=1.e8), extent=[res['xrange'][0], res['xrange'][1], res['yrange'][0], res['yrange'][1]])
plt.imshow(res['map'].transpose(), origin='lower', extent=[res['xrange'][0], res['xrange'][1], res['yrange'][0], res['yrange'][1]])
plt.show()


