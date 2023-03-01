import numpy as np
from matplotlib.colors import LogNorm
import pygadgetreader as pygr
from proj2d.makemap import makemap
import matplotlib.pyplot as plt
import time as t

indir = '/Users/mauro/XRISM/TheThreeHundred/Gadget3PESPH/NewMDCLUSTER_0322/'
infile = 'snap_128'

mass_snap = sum(pygr.readsnap(indir+infile, 'mass', 'gas', units=0)) # [10^10 h^-1 M_Sun]

start = t.time()
res = makemap(indir + infile, 'rho', struct = True, npix=512)
#res = makemap(indir + infile, 'wew', struct = True, npix=1024, center=[500e3, 500e3], size=5e3, nosmooth=True)#, tcut=1.e5)
end = t.time()
mass_map = sum(sum(res['map'])) * res['pixel_size'] ** 2
print('Mass snapshot = ', mass_snap)
print('Mass map = ', mass_map)
vmin = res['map'][np.where(res['map'] > 0.)].min()
print(res['map'].min(), vmin, mass_map, res['map'].max(), end-start)

#plt.imshow(res, norm=LogNorm(vmin=1.e6, vmax=1.e8))
#plt.imshow(res['map'], norm=LogNorm(vmin=1.e6, vmax=1.e8), extent=[res['xrange'][0], res['xrange'][1], res['yrange'][0], res['yrange'][1]])
#plt.imshow(res['map'].transpose(), norm=LogNorm(vmin=res['map'].min(), vmax=res['map'].max()), origin='lower', extent=[res['xrange'][0], res['xrange'][1], res['yrange'][0], res['yrange'][1]])
#plt.imshow(res['map'].transpose(), norm=LogNorm(vmin=vmin, vmax=res['map'].max()), origin='lower', extent=[res['xrange'][0], res['xrange'][1], res['yrange'][0], res['yrange'][1]])
plt.imshow(res['map'].transpose(), origin='lower', extent=[res['xrange'][0], res['xrange'][1], res['yrange'][0], res['yrange'][1]])
plt.show()


