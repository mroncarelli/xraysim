import xspec as xsp
xsp.Xset.chatter = 0

def bapec(spectrum, erange=None, start=None, fixed=None, method='cstat', niterations=100, criticaldelta = 1.e-3):
    return None