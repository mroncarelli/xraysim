import numpy as np
from proj2d.makeMap import makeMap
from proj2d.intKernel import intKernel

indir = '/Users/mauro/XRISM/TheThreeHundred/Gadget3PESPH/NewMDCLUSTER_0322/'
infile = 'snap_128'

res = makeMap(indir+infile)