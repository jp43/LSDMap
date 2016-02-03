# x2h: stripped-down version of the MDPlus Cofasu library.
import numpy as np
import mdtraj as mdt
import tempfile
import os
import h5py
import dask.bag as db
import dask.array as da
CHUNKS = 10000

def pib(coords, box):
    '''
    Pack coordinates into periodic box
    '''

    assert len(coords.shape) == 2
    assert coords.shape[1] == 3
    assert box.shape == (3, 3)

    boxinv = np.zeros((3))
    boxinv[0] = 1.0 / box[0,0]
    boxinv[1] = 1.0 / box[1,1]
    boxinv[2] = 1.0 / box[2,2]

    for xyz in coords:
        s = np.floor(xyz[2] * boxinv[2])
        xyz[2] -= s * box[2,2]
        xyz[1] -= s * box[2,1]
        xyz[0] -= s * box[2,0]

        s = np.floor(xyz[1] * boxinv[1])
        xyz[1] -= s * box[1,1]
        xyz[0] -= s * box[1,0]

        s = np.floor(xyz[0] * boxinv[0])
        xyz[0] -= s * box[0,0]

    return coords

class Fasu:
    def __init__(self, topology, trajectory, frames=None, selection='all', 
    centre=None, pack_into_box=False):
        '''
        A Fasu defines the required data to be extracted from one or more
        trajectory files.

        Arguments are:
        topology:   
            name of a topology file compliant with the trajectory files(s).
        trajectory: 
            list of one or more trajectory files.
        frames:     
            selection of trajectory frames to include. Can be a slice object, 
            or a numpy array.
        selection:  
            MDTraj-compliant atom selection string.
        centre:     
            If not None, an MDTraj-compliant atom selection string that defines
            atoms whose geometric centre will be moved to the centre of the 
            periodic box (if there is one), or to the origin (if there isn't).
        pack_into_box: 
            if True, all coordinates will be imaged into the primary unit cell,
            after any centering has been done.

        '''
        test = open(topology, 'r')
        test.close()

        self.topology = topology
        self.trajectory = trajectory
        if not isinstance(trajectory, list):
            self.trajectory = [trajectory,]

        for t in self.trajectory:
            test = open(t, 'r')
            test.close()

        if frames is not None:
            if not (isinstance(frames, slice) or isinstance(frames, np.ndarray)):
                raise TypeError('frames must be a slice object or numpy array')

        self.frames = frames
        self.selection = selection
        self.centre = centre
        self.pack_into_box = pack_into_box

    def _process(self, hdfdir=None):
        '''
        Private function that processes the Fasu definition, returning the name
        of an HDF5-format file that contains the required data.

        Arguments:
            hdfdir: 
            The name of the directory in which the HDF5 format file
            will be created. If not given, the OS-defined temporary directory
            is used.
        '''

        u = mdt.load(self.topology)
        sel = u.top.select(self.selection)
        ext = os.path.splitext(self.trajectory[0])[1].lower()
        if not ext in ['.gro', '.pdb']:
            u = mdt.load(self.trajectory, top=self.topology, atom_indices=sel)
        else:
            u = mdt.load(self.trajectory, atom_indices=sel)

        masses = [atom.element.mass for atom in u.top.atoms]
        masses = np.array(masses, dtype='float32')
        names = [u.top.atom(i).name for i in range(u.n_atoms)]

        h5filename = tempfile.NamedTemporaryFile(dir=hdfdir).name

        if self.frames is not None:
            x = np.array(u.xyz[self.frames], dtype='float32')
        else:
            x = np.array(u.xyz, dtype='float32')

        if self.centre is not None:
            c = u.top.select(self.centre)
            for i in range(len(x)):
                cx = x[i][c].mean(axis=0)
                if u.unitcell_vectors is None:
                    shift = -cx
                else:
                    shift = u.unitcell_vectors[i].diagonal()/2 - cx
                x[i] = x[i] + shift

        if self.pack_into_box:
            for i in range(len(x)):
                x[i] = pib(x[i], u.unitcell_vectors[i])

        hf = h5py.File(h5filename, 'w')
        hx = hf.create_dataset('x', data=x*10.0)
        hm = hf.create_dataset('m', data=masses)
        if u.unitcell_vectors is not None:
            hb = hf.create_dataset('b', data=u.unitcell_vectors)
        dt = h5py.special_dtype(vlen=unicode)
        hn = hf.create_dataset('n', data=names, dtype=dt)

        hf.close()
        self.top = u.top
        return h5filename

class Cofasu:
    '''
    A collection of Fasus.

    A Cofasu is created from a list of one or more Fasus:
        c = Cofasu(fasulist)
    or alternatively from a list of already-created HDF5-format files:
        c = Cofasu(hdf5filelist)

    The concatenated trajectories are then available as if in a numpy array:
        nframes, natoms, ndims = c.shape # ndims will always be 3
        frame0 = c[0]
    '''

    def __init__(self, fasulist, check=None, hdfdir=None):

        if not isinstance(fasulist, list):
            fasulist = [fasulist,]

        if isinstance(fasulist[0], str):
            self.hflist = fasulist
        else:
            b = db.from_sequence([f._process(hdfdir=hdfdir) for f in fasulist])
            self.hflist = b.compute()

        self.hlist = [h5py.File(h, 'r+') for h in self.hflist]
        
        if check is "names":
            nref = self.hlist[0]['n']
            for i in range(1, len(self.hlist)):
                if not (nref[:] == self.hlist[i]['n'][:]).all:
                    raise ValueError('Fasus with mismatched atom names')

        elif check is "masses":
            mref = self.hlist[0]['m']
            for i in range(1, len(self.hlist)):
                if not (mref[:] == self.hlist[i]['m'][:]).all:
                    raise ValueError('Fasus with mismatched atom masses')

        xs = [da.from_array(h['x'], chunks=CHUNKS) for h in self.hlist]
        self.x = da.concatenate(xs)
        if 'h' in h:
            xb = [da.from_array(h['b'], chunks=CHUNKS) for h in self.hlist]
            self.box = da.concatenate(xb)
        else:
            self.box = None
        self.fasulist = fasulist
        self.masses = self.hlist[0]['m']
        self.shape = self.shape()
        self.top = self.fasulist[0].top

    def __len__(self):
        return len(self.x)

    def __getitem__(self, key):
        return self.x[key].compute()

    def shape(self):
        return self.x.shape

    def write(self, filename, coordinates=None):
        '''
        Writes selected data to an output file, of format specified by the
        given filename's extension.

        Arguments:
            filename:
            Name of the file to be written. All MDTraj-supported formats are
            available.

            coordinates:
            An [nframes, natoms, 3] numpy array defining what will be written,
            else all frames in the Cofasu will be output.
        '''

        # Note: currently ignores box data.
        ext = os.path.splitext(filename)[1].lower()
        needs_topology = ext in ['.gro', '.pdb']

        if coordinates is None:
            coordinates = self.x.compute()

        with mdt.open(filename, 'w') as f:
            if needs_topology:
                f.write(coordinates, self.top)
            else:
                f.write(coordinates)
