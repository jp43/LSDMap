import os
import sys
import numpy as np
import argparse
import ConfigParser
import pickle
import random
import logging
import gc
from lsdmap.rw import x2h
#from MDPlus.core import cofasu

from lsdmap.rw import reader
from lsdmap.rw import coord_reader
from lsdmap.mpi import p_arpack
from lsdmap.mpi import p_index
from lsdmap.util import metric as mt

from mpi4py import MPI

class LSDMap(object):

    def initialize(self, comm, config, args):

        rank = comm.Get_rank()
        size = comm.Get_size()

        self.config = config
        self.args = args

        filename = args.struct_file[0] 
        self.struct_filename = filename

        if rank == 0:
            f = x2h.Fasu(args.topfile, args.struct_file, selection=args.selection)
            c = x2h.Cofasu(f)
            self.coords = np.array(c.x, dtype=np.double) * 0.1
            self.coords = np.swapaxes(self.coords, 1, 2) 
        else: 
            self.coords = None
        self.coords = comm.bcast(self.coords, root=0)

        print rank, self.coords.shape
        self.npoints = self.coords.shape[0]
        self.natoms = self.coords.shape[2]
        self.idxs_thread, self.npoints_per_thread, self.offsets_per_thread = p_index.get_idxs_thread(comm, self.npoints)

        logging.info('input coordinates loaded')

        self.initialize_local_scale()
        self.initialize_weights()
        self.initialize_metric()

    def initialize_local_scale(self):

        config = self.config
        args = self.args

        known_status = ['constant', 'kneighbor', 'user', 'kneighbor_mean', 'kneighbor_iw']
        _mapped = {'const': 'constant', 'cst': 'constant', 'mean-kneighbor': 'mean_kneighbor'}

        if args.epsfile is None:
            status = config.get('LOCALSCALE','status')
            if status in _mapped:
                status = _mapped[status]
            if not status in known_status:
                raise ValueError("local scale status should be one of "+ ', '.join(known_status))
            if status in ['kneighbor', 'kneighbor_mean', 'kneighbor_iw']:
                value = None
                self.k = config.getint('LOCALSCALE', 'k')
            if status == 'constant':
                value = config.getfloat('LOCALSCALE', 'epsilon')
                value = value*np.ones(self.npoints)
            if status == 'user':
                raise NameError("status 'user' specified in configuration file but epsfile not provided. Please consider option flag -e")
        else:
            espfile = reader.open(epsfile)
            status ='user'
            value = epsfile.readlines()

        self.status_epsilon = status
        self.epsilon = value


    def initialize_weights(self):

        config = self.config
        args = self.args

        if args.wfile is None:
            self.weights = np.ones(self.npoints, dtype='float')
        else:
            if os.path.isfile(args.wfile):
                wfile = reader.open(args.wfile)
                weights = wfile.readlines()
                self.weights = self.npoints*weights/np.sum(weights)
            else:
                logging.warning('.w file does not exist, set weights to 1.0.')
                self.weights = np.ones(self.npoints, dtype='float')

    def initialize_metric(self):

        _known_prms = ['r0']

        config = self.config
        self.metric = config.get('LSDMAP','metric')
        if config.has_option('LSDMAP','weight_method'):
          self.weight_method=config.get('LSDMAP','weight_method')
        else:
          self.weight_method='standard'
        
        if config.has_option('LSDMAP','n_eg'):
          self.neigs = config.getint('LSDMAP','n_eg')
        else:
          self.neigs = 10
         
        self.metric_prms = {}
        for prm in _known_prms:
            try:
                self.metric_prms[prm] = config.getfloat('LSDMAP', prm)
            except:
                pass

    def create_arg_parser(self):

        parser = argparse.ArgumentParser(description="Run LSDMap..") # message displayed when typing lsdmap -h

        # required options
        parser.add_argument("-f",
            type=str,
            dest="config_file",
            help='Configuration file (input): ini')

        parser.add_argument("-t",
            type=str,
            dest="topfile",
            help='Topology file (input): gro, pdb')

        parser.add_argument("-c",
            type=str,
            dest="struct_file",
            nargs='*',
            help = 'Structure file (input): gro, xvg')

        # other options
        parser.add_argument("-o",
            type=str,
            dest="output_file",
            help='Filename to save pickled LSDMap object (output, opt.): p')

        parser.add_argument("-s",
            type=str,
            dest="selection",
            default="all",
            help="Atoms selection for analysis (input, opt.)")

        parser.add_argument("-d",
            type=str,
            dest="dmfile",
            help="Filename to save distance matrix (output, opt.): .dm")

        parser.add_argument("-n",
            type=str,
            dest="nnfile",
            help='Filename to save the indices of the nearest neighbors of all configurations\
                  within the structure file (output, opt.): nn.\n When --ns or --nc flags are not\
                  specified: store only the neighbors within the range of the local scale.')

        parser.add_argument("-D",
            type=str,
            dest="dminput",
            help="Filename to load distance matrix from (input, opt.): .dm")

        parser.add_argument("-w",
            type=str,
            dest="wfile",
            help='File containing the weights of every point in a row (input, opt.): .w')

        parser.add_argument("-e",
            type=str,
            dest="epsfile",
            help='File containing the local scales of every point in a row (input, opt.): .eps')

        parser.add_argument("--ns",
            type=int,
            dest="nneighbors",
            help="Number of nearest neighbors stored in .nn file; should be used with option -n")

        parser.add_argument("--nc",
            type=float,
            dest="nneighbors_cutoff",
            help="Only the indices of neighbors closer than a distance of nneighbors_cutoff are stored in .nn file; should be used with option -n")
        
        parser.add_argument("-V", "--version", action="store_true", default=False, help="display verison and exit") 
        
        return parser


    def compute_kernel(self, comm, npoints_thread, distance_matrix_thread, weights_thread, epsilon_thread,weight_method='standard'):
        # for a detailed description of the following operations, see the paper:
            # Determination of reaction coordinates via locally scaled diffusion map
            # Mary A. Rohrdanz, Wenwei Zheng, Mauro Maggioni, and Cecilia Clementi
            # The Journal of Chemical Physics 134, 124116 (2011)

      if weight_method=='standard': 
        p_vector_thread = np.zeros(npoints_thread, dtype='float')
        d_vector_thread = np.zeros(npoints_thread, dtype='float')

        # compute LSDMap kernel, Eq. (5) of the above paper
        kernel = np.sqrt((weights_thread[:, np.newaxis]).dot(self.weights[np.newaxis])) * \
                 np.exp(-distance_matrix_thread**2/(2*epsilon_thread[:, np.newaxis].dot(self.epsilon[np.newaxis])))

        
        p_vector_thread = np.sum(kernel, axis=1)
        p_vector = np.hstack(comm.allgather(p_vector_thread)) # Eq. (6)
        self.p_vector = p_vector
        #print kernel.shape, p_vector_thread.shape, weights_thread.shape, self.weights.shape

        kernel /= np.sqrt(p_vector_thread[:,np.newaxis].dot(p_vector[np.newaxis])) # Eq. (7)
        d_vector_thread = np.sum(kernel, axis=1)
        d_vector = np.hstack(comm.allgather(d_vector_thread)) # Compute D given between Eqs. (7) and (8)
        self.d_vector = d_vector

        kernel /= np.sqrt(d_vector_thread[:,np.newaxis].dot(d_vector[np.newaxis])) # Eq (8) (slightly modified)
      else:
        p_vector_thread = np.zeros(npoints_thread, dtype='float')
        d_vector_thread = np.zeros(npoints_thread, dtype='float')

        # compute LSDMap kernel, Eq. (5) of the above paper

        kernel = np.exp(-distance_matrix_thread**2/(2*epsilon_thread[:, np.newaxis].dot(self.epsilon[np.newaxis])))
        
        for i in range(npoints_thread):
            p_vector_thread[i]= np.dot(kernel[i,:], self.weights)

        #p_vector_thread = np.sum(kernel, axis=1)
        p_vector = np.hstack(comm.allgather(p_vector_thread)) # Eq. (6)
        self.p_vector = p_vector
        #print "new method"
        #print kernel.shape, p_vector_thread.shape, weights_thread.shape, self.weights.shape


        kernel /= np.sqrt(p_vector_thread[:,np.newaxis].dot(p_vector[np.newaxis])) # Eq. (7)
        #d_vector_thread = np.sum(kernel, axis=1)

        for i in range(npoints_thread):
            d_vector_thread[i]= np.dot(kernel[i,:], self.weights)

        d_vector = np.hstack(comm.allgather(d_vector_thread)) # Compute D given between Eqs. (7) and (8)
        self.d_vector = d_vector

        for i in range(npoints_thread):
            kernel[i,:]= kernel[i,:]*self.weights

        kernel /= np.sqrt(d_vector_thread[:,np.newaxis].dot(d_vector[np.newaxis])) # Eq (8) (slightly modified)
        
      return kernel


    def save(self, config, args):
        """
        save LSDMap object in .lsdmap file and eigenvalues/eigenvectors in .eg/.ev files
        """

        if isinstance(self.struct_filename, list):
            struct_filename = self.struct_filename[0]
        else:
            struct_filename = self.struct_filename

        path, ext = os.path.splitext(struct_filename)
        np.savetxt(path + '.eg', np.fliplr(self.eigs[np.newaxis]), fmt='%9.6f')
        logging.info("saved .eg file")
        np.savetxt(path + '.ev', np.fliplr(self.evs), fmt='%.18e')
        logging.info("saved .ev file")
        np.savetxt(path + '.eps', np.fliplr(self.epsilon[np.newaxis]), fmt='%9.6f')
        logging.info("saved .eps file")
        if config.has_option('LSDMAP','print_kernel'):
          if config.get('LSDMAP','print_kernel')=='true': 
            np.savetxt(path + '.kernel', self.kernel, fmt='%.18e')
            logging.info("saved .kernel file")
        #np.save(path + '_eg.npy', np.fliplr(self.eigs[np.newaxis]))
        #np.save(path + '_ev.npy', np.fliplr(self.evs))

        if args.output_file is None:
            try:
                lsdmap_filename = config.get('LSDMAP', 'lsdmfile')
            except:
                return
        else:
            lsdmap_filename = args.output_file
        logging.info("checked args.output_file")
        with open(lsdmap_filename, "w") as file:
            pickle.dump(self, file)
        logging.info("pickle dumped")

    def save_nneighbors(self, comm, args, neighbor_matrix, idx_neighbor_matrix, epsilon_thread):

        size = comm.Get_size()  # number of threads
        rank = comm.Get_rank()  # number of the current thread
        if rank == 0:
            try:
                os.remove(args.nnfile)
            except OSError:
                pass
            nnfile = open(args.nnfile, 'a')
            for idx in xrange(size):
                if idx == 0:
                    epsilon = epsilon_thread
                else:
                    idx_neighbor_matrix = comm.recv(source=idx, tag=idx)
                    neighbor_matrix = comm.recv(source=idx, tag=1000+idx)
                    epsilon = comm.recv(source=idx, tag=2000+idx)
                if args.nneighbors is not None: # save first "args.nneighbors" nearest neighbors
                    for idxs in idx_neighbor_matrix:
                        np.savetxt(nnfile, idxs[:min(args.nneighbors,self.npoints)][np.newaxis], fmt='%d')
                elif args.nneighbors_cutoff is not None: # save all nearest neighbors within "args.nneighbors_cutoff"
                    for idxrow, row in enumerate(neighbor_matrix):
                        for idxcol, col in enumerate(row):
                            if col > args.nneighbors_cutoff:
                                break
                        np.savetxt(nnfile, idx_neighbor_matrix[idxrow,:idxcol][np.newaxis], fmt='%d')
                else: # save all nearest neighbors within the range of the local scale
                    for idxrow, row in enumerate(neighbor_matrix):
                        for idxcol, col in enumerate(row):
                            if col > epsilon[idxrow]:
                                break
                        np.savetxt(nnfile, idx_neighbor_matrix[idxrow,:idxcol][np.newaxis], fmt='%d')
            nnfile.close()
        else:
           comm.send(idx_neighbor_matrix, dest=0, tag=rank)
           comm.send(neighbor_matrix, dest=0, tag=1000+rank)
           comm.send(epsilon_thread, dest=0, tag=2000+rank)

    def load_distance_matrix(self, comm, args):
        size = comm.Get_size()  # number of threads
        rank = comm.Get_rank()  # number of the current thread
        # Load matrix chunk and send it to processor
        # Reading from binary format requires as many loads as there were saves.
        # Alternative, could read entire matrix in break it into chunks 
        # appropriate for the num of processors used currently
        if rank == 0:
            try:
                dminput = open(args.dminput, 'rb')
            except IOError:
                raise IOError("The inputted distance matrix file %s does not exist!" % args.dminput)
            for idx in range(size):
                distance_matrix = np.load(dminput)
                if idx == 0:
                    distance_matrix_thread = distance_matrix
                else:
                    comm.send(distance_matrix, dest=idx, tag=idx)
            dminput.close()
        else:
            distance_matrix_thread = comm.recv(source=0, tag=rank)
        return distance_matrix_thread
         
    def save_distance_matrix(self, comm, args, distance_matrix_thread):
        """ Save distance matrix as binary numpy file """ 
        size = comm.Get_size()  # number of threads
        rank = comm.Get_rank()  # number of the current thread
        if rank == 0:
            try:
                os.remove(args.dmfile)
            except OSError:
                pass
            dmfile = open(args.dmfile, 'a')
            for idx in xrange(size):
                if idx == 0:
                    distance_matrix = distance_matrix_thread
                else:
                    distance_matrix = comm.recv(source=idx, tag=idx)
                np.savetxt(dmfile, distance_matrix)
            dmfile.close()
        else:
            comm.send(distance_matrix_thread, dest=0, tag=rank)

    def run(self):

        #initialize mpi variables
        comm = MPI.COMM_WORLD   # MPI environment
        size = comm.Get_size()  # number of threads
        rank = comm.Get_rank()  # number of the current thread 

        parser = self.create_arg_parser()
        args = parser.parse_args() # set argument parser
        
        if args.version: # Print version and exit
            if rank == 0:
                print "2.4"
            return
        else:
            if args.config_file == None:
                raise IOError("lsdmap option -f is required")
            if args.struct_file == None:
                raise IOError("lsdmap option -c is required") 
        
        config = ConfigParser.SafeConfigParser()
        config.read(args.config_file) # set config file parser

        logging.basicConfig(filename='lsdmap.log',
                            filemode='w',
                            format="%(levelname)s:%(name)s:%(asctime)s: %(message)s",
                            datefmt="%H:%M:%S",
                            level=logging.DEBUG)


        logging.info('intializing LSDMap with %d processors...' % size)
        self.initialize(comm, config, args)
        logging.info('LSDMap initialized')

        if size > self.npoints:
            logging.error("number of threads should be less than the number of frames")
            raise ValueError

        npoints_thread = self.npoints_per_thread[rank]
        coords_thread = np.array([self.coords[idx] for idx in self.idxs_thread])
        weights_thread = np.array([self.weights[idx] for idx in self.idxs_thread])

        if args.dminput is None:
            # compute the distance matrix
            DistanceMatrix = mt.DistanceMatrix(coords_thread, self.coords, metric=self.metric, metric_prms=self.metric_prms)
            distance_matrix_thread = DistanceMatrix.distance_matrix
            neighbor_matrix_thread, idx_neighbor_matrix_thread = DistanceMatrix.get_neighbor_matrix()
            logging.info("distance matrix computed")
        else:
            # load distance matrix
            distance_matrix_thread = self.load_distance_matrix(comm,args)
            neighbor_matrix_thread, idx_neighbor_matrix_thread = mt.get_neighbor_matrix(distance_matrix_thread)
            logging.info("distance matrix loaded")

        # compute kth neighbor local scales if needed
        if self.status_epsilon in ['kneighbor', 'kneighbor_mean', 'kneighbor_iw']:
            #epsilon_thread = []
            epsilon_threadv = np.zeros(npoints_thread,dtype='float')
            for idx, line in enumerate(idx_neighbor_matrix_thread):
                cum_weight = 0
                for jdx in line[1:]:
                    if self.status_epsilon == 'kneighbor_iw':
                        cum_weight += 1
                    else:
                        cum_weight += self.weights[jdx]
                    if cum_weight >= self.k:
                        break
                #epsilon_thread.append(distance_matrix_thread[idx,jdx])
                epsilon_threadv[idx] = distance_matrix_thread[idx,jdx]

            self.epsilon = np.zeros(self.npoints, dtype='float')
            comm.Allgatherv(epsilon_threadv, (self.epsilon, self.npoints_per_thread, self.offsets_per_thread, MPI.DOUBLE))

            if self.status_epsilon == 'kneighbor_mean':
                mean_value_epsilon = np.mean(self.epsilon) # compute the mean value of the local scales
                self.epsilon = mean_value_epsilon * np.ones(self.npoints)  # and set it as the new constant local scale

            logging.info("kneighbor local scales computed")
            
        epsilon_thread = np.array([self.epsilon[idx] for idx in self.idxs_thread])

        # compute kernel
        kernel = self.compute_kernel(comm, npoints_thread, distance_matrix_thread, weights_thread, epsilon_thread,weight_method=self.weight_method)

        self.kernel=kernel
        # diagonalize kernel
        params= p_arpack._ParallelSymmetricArpackParams(comm, kernel, self.neigs)
        while not params.converged:
            params.iterate()
        eigs, evs = params.extract(return_eigenvectors=True)

        # normalize eigenvectors
        self.evsu = np.copy(evs) # store unormalized eigenvectors
        evs /= np.sqrt(self.d_vector[:,np.newaxis])
        norm = np.sqrt(np.sum(evs**2, axis=0))
        evs /= norm[np.newaxis,:]
        

        # store eigenvalues/eigenvectors
        self.eigs = eigs
        self.evs = evs

        logging.info("kernel diagonalized")
    
        if rank == 0:    
            self.save(config, args)
        logging.info("Eigenvalues/eigenvectors saved (.eg/.ev files)")

        # store nearest neighbors in .nn file if specified via -n option
        if args.nnfile is not None:
            logging.info("Saving nearest neighbors")
            self.save_nneighbors(comm, args, neighbor_matrix_thread, idx_neighbor_matrix_thread, epsilon_thread)

        if (args.dmfile is not None) and (args.dminput is None):
            logging.info("Saving distance matrix")
            self.save_distance_matrix(comm, args, distance_matrix_thread)

        logging.info("LSDMap computation done")

if __name__ == '__main__':
    LSDMap().run()
