from src.pkg.specutils.sixte import make_pha
import os
import glob

indir = os.environ.get('HOME') + '/XRISM/Evlist/'
outdir = os.environ.get('HOME') + '/XRISM/Pha/'
fileList = glob.glob(indir + '*.evt')

for evtFile in fileList:
    fileName = os.path.basename(evtFile)
    phaFile = outdir + fileName.replace('.evt', '.pha')
    logFile = outdir + fileName.replace('.evt', '.log')
    if not os.path.isfile(phaFile):
        sys_out = make_pha(evtFile, phaFile, logfile=logFile)
        if sys_out == 0:
            print("Event list saved in " + phaFile)
        else:
            print("ERROR - System output: ", sys_out)
