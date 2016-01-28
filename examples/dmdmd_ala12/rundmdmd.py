from wrappers import grompp, mdrun, lsdmap, selection, reweighting
from wrappers.tools import pack
import dask.bag as db
import shutil

start = 'start.gro'
topol = 'topol.top'
mdp = 'grompp.mdp'
lsdmap_config = 'lsdmap.ini'
atom_selection = "'not element == H'"
nreps = 50
ncycles = 5

grompp.DEFAULTS['-f'] = mdp
grompp.DEFAULTS['-p'] = topol
lsdmap.DEFAULTS[0] = 'mpirun -n 2 lsdmap'
lsdmap.DEFAULTS['-f'] = lsdmap_config
lsdmap.DEFAULTS['-s'] = atom_selection

startfiles = [start,] * nreps
wfile = None

for cycle in range(ncycles):
    ginps = []
    for i in range(nreps):
        ginp = grompp.new_inputs()
        ginp['-c'] = startfiles[i]
        ginps.append(ginp)

    b = db.from_sequence(ginps).map(grompp.run)
    print 'Starting grompp runs...'
    gouts = b.compute()

    minps = []
    for i in range(nreps):
        minp = mdrun.new_inputs()
        minp['-s'] = gouts[i]['-o']
        minps.append(minp)

    b = db.from_sequence(minps).map(mdrun.run)
    print 'Starting MD jobs...'
    mouts = b.compute()

    crdfiles = [mout['-c'] for mout in mouts]
    crdlist = pack.pack(crdfiles)

    print 'Running LSDMap...'
    linp = lsdmap.new_inputs()
    linp['-c'] = crdfiles
    linp['-t'] = start
    if wfile is not None:
        linp['-w'] = wfile

    lout = lsdmap.run(linp)

    shutil.copy(lout['-ev'], './results/iter{}.ev'.format(cycle))

    print 'Running selection...'
    sinp = selection.new_inputs()
    sinp['-s'] = lout['-ev']
    sinp[1] = nreps

    sout = selection.run(sinp)

    print 'Running reweighting...'
    rinp = reweighting.new_inputs()
    rinp['-c'] = crdlist
    rinp['-n'] = lout['-n']
    rinp['-s'] = sout['-o']

    rout = reweighting.run(rinp)

    startfiles = pack.unpack(rout['-o'])
    wfile = rout['-w']
