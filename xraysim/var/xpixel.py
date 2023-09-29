from xfit import XFit

class XPixel:
    sim_type = None
    sim_file = None
    xrange = None
    yrange = None
    zrange = None
    ra = None
    dec = None
    norm = None
    tmw = None
    tew = None
    tsl = None
    abund = None
    z = None
    vmw = None
    vew = None
    wmw = None
    wew = None
    fitRes = None

    def add_fitresult(self, fit_result: XFit):
        if self.fitRes is None:
            self.fitRes = fit_result
        elif type(self.fitRes) == XFit:
            self.fitRes = [self.fitRes, fit_result]
        else:
            self.fitRes.append(fit_result)

    def get_fitresult(self, index=0):
        if type(self.fitRes) == list:
            return self.fitRes[index]
        else:
            return self.fitRes
