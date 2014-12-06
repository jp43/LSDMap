import os
import sys
import time
import imp
import argparse
import subprocess
import ConfigParser

from math import sqrt, floor
import logging
import radical.pilot
import numpy as np

from dmaps.tools import pilot
from dmaps.tools.config import platforms
from dmaps.kernel import worker

class DMapSampling(object):

    def initialize(self, settings):

        if settings.startgro == 'input.gro':
            raise ValueError("name input.gro is already allocated for md input file and thus can not be assigned to variable startgro! \
	                      Please use a different name")

        if hasattr(settings, "inifile"):
            inifile = settings.inifile
        else:
            inifile = "config.ini"
            settings.inifile = inifile

        if not os.path.isfile(inifile):
            logging.error(".ini file does not exist:" + inifile)
            raise IOError(".ini file does not exist:" + inifile)

        if settings.iter == 0:
            with open('input.gro', 'w') as ifile:
                for idx in xrange(settings.nreplicas):
                    with open(settings.startgro, 'r') as sfile:
                       for line in sfile:
                          print >> ifile, line.replace("\n", "")
            os.system("sed -i" + self.sedarg + "'s/isfirst=.*/isfirst=1/g' " + settings.inifile)
        else:
            os.system("sed -i" + self.sedarg + "'s/isfirst=.*/isfirst=0/g' " + settings.inifile)

        # load parameters inside inifile
        self.load_parameters(settings)

        # create executables
        self.write_md_script("run_md.sh", settings)

    def load_parameters(self, settings):

        config = ConfigParser.SafeConfigParser()
        config.read(settings.inifile)

        if hasattr(settings, "mdpfile"):
            mdpfile = settings.mdpfile
        else:
            mdpfile = "grompp.mdp"
            settings.mdpfile = mdpfile

        if not os.path.isfile(mdpfile):
            logging.error(".mdp file does not exist:" + mdpfile)
            raise IOError(".mdp file does not exist:" + mdpfile)

        # number of configurations saved per replica
        self.nstride = config.getint('DMAPS', 'nstride')

        # set number of steps in MD simulations
        self.nsteps_min = 10000

        if config.has_option('MD', 'nsteps'):
            self.nsteps = config.getint('MD', 'nsteps')
            os.system("sed -i" + self.sedarg + "'s/nsteps.*/nsteps                   = %i/g' "%self.nsteps + settings.mdpfile)
            logging.info("nsteps for MD simulations was set to %i "%self.nsteps)
            self.nsteps_guess = False
        else:
            if settings.iter == 0:
                # if first iteration, the number of steps is set to 10000 or so
                self.nsteps = self.nstride*(self.nsteps_min/self.nstride)
                # update nsteps line in .mdp file
                os.system("sed -i" + self.sedarg + "'s/nsteps.*/nsteps                   = %i/g' "%self.nsteps + settings.mdpfile)
                logging.info("nsteps for MD simulations was set to %i (default value)"%self.nsteps)
            else:
                self.nsteps = int(subprocess.check_output("cat " + settings.mdpfile + " | sed -n -e 's/^.*nsteps.*=//p' | tr -d ' '", shell=True))
            self.nsteps_guess = True

        # first iteration?
        self.isfirst = config.getint('DMAPS', 'isfirst')

        # number of first dcs used
        self.ndcs = config.getint('DMAPS', 'ndcs')

        # total number of configurations
        self.npoints = settings.nreplicas * self.nstride

        # number of configurations used to compute lsdmap
        nlsdmap = config.getint('LSDMAP', 'nlsdmap')
        if nlsdmap > self.npoints:
            logging.warning("number of configs required for LSDMap (%i) is larger than the total number of configs expected every iteration (%i)"%(nlsdmap, self.npoints))
            logging.warning("set the number of configs used for LSDMap to %i" %self.npoints)
            self.nlsdmap = self.npoints
        else:
            self.nlsdmap = nlsdmap

        # number of bins used to build the histogram for the free energy
        self.nbinsfe = int(sqrt(self.npoints))

        # temperature in Kelvins
        temperature = config.getint('MD', 'temperature')
        kb = 8.31451070e-3 #kJ.mol^(-1).K^(-1)
        self.kT = kb*temperature

        try:
            # solvation?
            self.solvation = config.get('MD','solvation').lower()
        except:
            self.solvation = 'none'

        if self.solvation in ['none', 'implicit']: 
            os.system("sed -i" + self.sedarg + "'s/ref_t.*/ref_t               = %i/g' "%temperature + settings.mdpfile)
        elif self.solvation == 'explicit':
            os.system("sed -i" + self.sedarg + "'s/ref_t.*/ref_t               = %i %i/g' "%(temperature, temperature) + settings.mdpfile)
        else:
	    raise ValueError("solvation option in configuration file not understood, should one among none, implicit and explicit")

        os.system("sed -i" + self.sedarg + "'s/gen_temp.*/gen_temp                = %i/g' "%temperature + settings.mdpfile)
 
        # cutoff when computing the free energy
        ncutoff =  config.getint('DMAPS', 'ncutoff')
        self.cutoff = ncutoff*self.kT

        # number of points used for the fitting of the dc values
        self.nfitdcs = config.getint('FITTING', 'npoints')

        # number of bins used to build the histogram to select the fitting points
        self.nbinsdcs = int(sqrt(self.nlsdmap))

        # number of bins used to build the histogram to select the lsdmap points
        self.nbins_lsdmap = int(sqrt(self.npoints))


    def write_md_script(self, filename, settings):

        inifile = settings.inifile
        mdpfile = settings.mdpfile

        # check topfile
        if hasattr(settings, "topfile"):
            topfile = settings.topfile
        else:
            topfile = "topol.top"
            settings.topfile = topfile

        if not os.path.isfile(topfile):
            logging.error(".top file does not exist:" + topfile)
            raise IOError(".top file does not exist:" + topfile)

        # check grompp and mdrun options
        if hasattr(settings, "grompp_options"):
            grompp_options = settings.grompp_options
        else:
            grompp_options = ""

        if hasattr(settings, "mdrun_options"):
            mdrun_options = settings.mdrun_options
        else:
            mdrun_options = ""

        # write script
        with open(filename, 'w') as file:
            script ="""#!/bin/bash

# this script was automatically created when using DMap Sampling

startgro=input.gro
tmpstartgro=tmp.gro

natoms=$(sed -n '2p' $startgro)
nlines_per_frame=$((natoms+3))

nlines=`wc -l $startgro| cut -d' ' -f1`
nframes=$((nlines/nlines_per_frame))

for idx in `seq 1 $nframes`; do

  start=$(($nlines_per_frame*(idx-1)+1))
  end=$(($nlines_per_frame*idx))
  sed "$start"','"$end"'!d' $startgro > $tmpstartgro

  # gromacs preprocessing & MD
  grompp -f %(mdpfile)s -c $tmpstartgro -p %(topfile)s %(grompp_options)s &> /dev/null
  mdrun -nt 1 -dms %(inifile)s -s topol.tpr %(mdrun_options)s &> mdrun.log

done

# remove temporary files
rm -f $tmpstartgro
        """ % locals()
            file.write(script)


    def update(self, args, settings):

        # before updating save data
        os.system('rm -rf iter%i'%settings.iter)
        os.system('mkdir iter%i'%settings.iter)
        os.system('cp ' + ' '.join(['confall.gro', 'confall.w', 'confall.ev.embed', 'output.gro', 'output.ev']) + ' iter%i'%settings.iter)
        os.system('cp -r ' + ' '.join(['lsdmap', 'fit', 'fe']) + ' iter%i'%settings.iter)
 
        if settings.iter > 0:
            os.system('cp confall.ev.embed.old iter%i'%settings.iter)

        autocorrelation_time_dcs = self.get_dcs_autocorrelation_time(settings)

        #if self.nsteps_guess:
        #    min_autocorrelation_time_dc = min(autocorrelation_time_dc1, autocorrelation_time_dc2)
        #    self.nsteps = max(self.nstride*min_autocorrelation_time_dc/settings.nreplicas, self.nstride*(self.nsteps_min/self.nstride))
        #    os.system("sed -i" + self.sedarg + "'s/nsteps.*/nsteps                   = %i/g' "%self.nsteps + settings.mdpfile)
        #    logging.info("nsteps for next MD simulations has been set to %i"%self.nsteps)

        os.system('mv output.gro input.gro')
        os.system('sed -i' + self.sedarg + "'s/iter=.*/iter=%i/g' "%(settings.iter+1) + args.setfile)
        os.system('sed -i' + self.sedarg + "'s/isfirst=.*/isfirst=0/g' " + settings.inifile)

        return settings.iter + 1


    def get_dcs_autocorrelation_time(self, settings):

        # if first iteration, compute autocorrelation time from computed dcs 
        if settings.iter == 0:
            os.system('cp confall.ev.embed autocorr.ev')

        dcs = np.loadtxt('autocorr.ev')
	if self.ndcs == 1:
	    dcs = dcs[:,np.newaxis]
        nvalues = dcs.shape[0]
        nvalues_per_replica = nvalues/settings.nreplicas
       
        step = np.linspace(0, self.nsteps, nvalues_per_replica+1).astype(int)
        step = step[:-1]
        autocorrelation_dcs = np.zeros((nvalues_per_replica, self.ndcs))
        autocorrelation_time_dcs = np.zeros(self.ndcs)

        logging.info("Compute DCs autocorrelation time")

        for idx in xrange(settings.nreplicas):
       
            # load dcs values
            dcs_values = dcs[idx*nvalues_per_replica:(idx+1)*nvalues_per_replica, :]
        
            vardcs = dcs_values.var(axis=0)
            meandcs = dcs_values.mean(axis=0)
            dcs_values -= meandcs

            for jdx in range(self.ndcs):
                tmp = np.correlate(dcs_values[:,jdx], dcs_values[:,jdx], mode='full')[-nvalues_per_replica:]
                tmp /= vardcs[jdx]*np.arange(nvalues_per_replica, 0, -1)
                autocorrelation_dcs[:,jdx] += tmp
        
        for jdx in xrange(self.ndcs):
            for idx in xrange(nvalues_per_replica):
                if autocorrelation_dcs[idx, jdx] <= autocorrelation_dcs[0, jdx]/2:
                    break
            autocorrelation_time_dcs[jdx] = step[idx]

        logging.info("Autocorrelation times (time steps): " + ", ".join(map(str,autocorrelation_time_dcs.tolist())))

        autocorrdir = 'autocorr'
        os.system('rm -rf autocorr; mkdir autocorr')

        np.savetxt('autocorr/autocorr.dat', np.hstack((step[:,np.newaxis], autocorrelation_dcs)), fmt='%15.7e')


    def restart_from_iter(self, num_iter, args):

        logging.info("restarting from iteration %i" %num_iter)
        # remove iter folders with iter number >= num_iter
        os.system('for dir in iter*; do num=`cut -d "r" -f 2 <<< $dir`; if [ "$num" -ge %i ]; then rm -rf $dir; fi ; done'%num_iter)
        os.system('cp iter%i/output.gro input.gro'%(num_iter-1))
        os.system('rm -rf ' + ' '.join(['lsdmap', 'fit', 'fe']))
        os.system('cp -r iter%i/{'%(num_iter-1) + ','.join(['lsdmap', 'fit', 'fe']) + '} .')

        # update iter in settings file
        os.system('sed -i' + self.sedarg + "'s/iter=.*/iter=%i/g' "%num_iter + args.setfile)

        return

    def restart(self, args):
       
        os.system('sed -i' + self.sedarg + "'s/iter=.*/iter=0/g' settings")
        os.system('rm -rf iter* ' + ' '.join(['lsdmap', 'fit', 'fe', 'md']))

        return

    def run(self):

        parser = argparse.ArgumentParser(description="Run Diffusion Map Driven Adaptive Sampling...")
        parser.add_argument("-f", type=str, dest="setfile", required=True, help='File containing settings (input): -')
        parser.add_argument("--restart", action="store_true", dest="restart", default=False, help='restart from scratch')
        parser.add_argument("--checkpoint", type=int, dest="num_iter", help='restart from a given iteration')

        args = parser.parse_args()

        logging.basicConfig(filename='dmaps.log',
                            filemode='w',
                            format="%(levelname)s:%(name)s:%(asctime)s: %(message)s",
                            datefmt="%H:%M:%S",
                            level=logging.DEBUG)

        if sys.platform in platforms['mac']:
            self.sedarg = " '' "
        else:
            self.sedarg = " "

        if args.restart:
            self.restart(args)
            if args.num_iter is not None:
	        logging.error("checkpoint option can not be set together with restart option")

        if args.num_iter is not None:
            if args.num_iter == 0:
                logging.error("checkpoint option can not be set to 0, use restart option instead")
            elif args.num_iter > 0:
                self.restart_from_iter(args.num_iter, args)
            else:
                logging.error("argument of checkpoint option should be a positive integer (iteration number to restart from)")

        settings = imp.load_source('setfile', args.setfile)
	umgr, session = pilot.startPilot(settings)

        # initialize dmap sampling
        self.initialize(settings)

        # main loop
        for idx in xrange(settings.niters):
            logging.info("START ITERATION %i"%settings.iter)
            # run biased MD
            dmapsworker = worker.DMapSamplingWorker()
            dmapsworker.run_md(umgr, settings)
            # run LSDMap
            dmapsworker.run_lsdmap(umgr,settings, self.npoints, self.nlsdmap, self.nbins_lsdmap, self.ndcs)
            # run fit
            dmapsworker.run_fit_dcs(umgr, settings, self.nfitdcs, self.nbinsdcs, self.ndcs)
            # estimate free energy (and pick new points)
            dmapsworker.do_free_energy(umgr, settings, self.nbinsfe, self.cutoff, self.kT, self.ndcs)
            # update for next iteration
            settings.iter = self.update(args, settings)
           
        session.close()
