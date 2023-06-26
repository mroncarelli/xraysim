from src.pkg.specutils.sixte import create_eventlist
import os

indir = os.environ.get('HOME') + '/XRISM/Simput/'
outdir = os.environ.get('HOME') + '/XRISM/Evlist/'
fileList = ['spcube_snap_128_5_']
instrument = 'athena-xifu'  # 'xrism-resolve'
t_exp = 1.e4  # [s]
exposure_tag = str(int(t_exp * 1e-3)).zfill(4) + 'ks'

for file in fileList:
    for proj in ['x', 'y', 'z']:
        infile = indir + file + proj + '_iso5.0_novel.simput'
        outfile = outdir + file + proj + '_' + instrument + '_' + exposure_tag + '_iso5.0_novel_nobkg.evt'
        logfile = outdir + file + proj + '_' + instrument + '_' + exposure_tag + '_iso5.0_novel_nobkg.log'
        sys_out = create_eventlist(infile, instrument, t_exp, outfile, background=False)
        if sys_out == 0:
            print("Event list saved in " + outfile)
        else:
            print("ERROR - System output: ", sys_out)
