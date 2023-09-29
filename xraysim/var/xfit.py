import numpy as np


class XFit:
    script = None
    norm: None
    t = None
    abund = None
    z = None
    w = None
    cint_norm: np.zeros(2)
    cint_t = np.zeros(2)
    cint_abund = np.zeros(2)
    cint_z = np.zeros(2)
    cint_w = np.zeros(2)
    nCounts = None
    nDF = None
    statType = None
    stat = None
