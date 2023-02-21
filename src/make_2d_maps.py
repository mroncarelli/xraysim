from matplotlib.colors import LogNorm

from proj2d.makemap import makemap
import matplotlib.pyplot as plt

indir = '/Users/mauro/XRISM/TheThreeHundred/Gadget3PESPH/NewMDCLUSTER_0322/'
infile = 'snap_128'

res, norm = makemap(indir + infile, 'rho')
print(res.min(), res.max())
plt.imshow(res)
plt.imshow(res)
plt.show()
