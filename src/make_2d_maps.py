from matplotlib.colors import LogNorm

from proj2d.makemap import makemap
import matplotlib.pyplot as plt

indir = '/Users/mauro/XRISM/TheThreeHundred/Gadget3PESPH/NewMDCLUSTER_0322/'
infile = 'snap_128'

res, norm = makemap(indir + infile, 'vmw')
print(res.min(), res.max())
#plt.imshow(res, norm=LogNorm(vmin=1.e6, vmax=1.e8))
plt.imshow(res)
plt.show()
