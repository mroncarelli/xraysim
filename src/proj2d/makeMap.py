import numpy as np
import pygadgetreader as pygr


def makeMap(fileName: str, center = None, size=None, qty='rho', proj='z', nPix=256):

    # Reading header variables
    hHubble = pygr.readhead(fileName, 'h')
    nGas = pygr.readhead(fileName, 'gascount')

    # Reading positions of particles
    pos = pygr.readsnap(fileName, 'pos', 'gas', units=1) # [h^-1 kpc] comoving
    if proj == 'x' or proj == 0:
        x = pos[:, 1]
        y = pos[:, 2]
    elif proj == 'y' or proj == 1:
        x = pos[:, 2]
        y = pos[:, 0]
    elif proj == 'z' or proj == 2:
        x = pos[:, 0]
        y = pos[:, 1]
    else:
        print("Invalid projection axis: ", proj, "Choose between 'x' (or 0), 'y' (1), 'z' (2)")
        raise ValueError
    del pos

    # Defining center and map size
    hsml = pygr.readsnap(fileName, 'hsml', 'gas', units=1) # [h^-1 kpc] comoving
    if center is None:
        xMin, xMax = min(x-hsml), max(x+hsml)
        yMin, yMax = min(y-hsml), max(y+hsml)
        xc = 0.5*(xMin+xMax)
        yc = 0.5*(yMin+yMax)
        if size == None:
            deltaX, deltaY = xMax-xMin, yMax-yMin
            if deltaX >= deltaY:
                size = deltaX
                x0, y0 = xMin, yMin-0.5*(deltaX-deltaY)
            else:
                size = deltaY
                x0, y0 = xMin-0.5*(deltaY-deltaX), yMin
        else:
            x0, y0 = xc-0.5*size, yc-0.5*size
    else:
        try:
            xc, yc = float(center[0]), float(center[1])
        except:
            print("Invalid center: ", center, "Must be a 2d number vector")
            raise ValueError

        if size == None:
            xMin, xMax = min(x-hsml), max(x+hsml)
            yMin, yMax = min(y-hsml), max(y+hsml)
            if not( xMin <= xc <= xMax and yMin <= yc <= yMax):
                print("WARNING: Map center is outside the simulation box")
                size = max(abs(xc-xMin), abs(xc-xMax), abs(yc-yMin), abs(yc-yMax))

        x0, y0 = xc-0.5*size, yc-0.5*size

    # Normalizing coordinates in pixel units (between 0 and nPix)
    x = (x-x0)/size*nPix
    y = (y-y0)/size*nPix
    hsml = hsml/size*nPix

    rho = pygr.readsnap(fileName, 'rho', 'gas', units=1)
    print(min(rho), max(rho))

    return 1