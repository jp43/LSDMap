"""Reader for .xvg files containing angles."""

import numpy as np
import itertools as it

pi = 3.14159265359

class XvgError(Exception):
    pass

class Reader(object):
    global pi

    def __init__(self, filename):

        if isinstance(filename, basestring):
            filename = [filename]
        else:
            filename = filename
        self.filename = filename
        self.nfiles = len(self.filename)
        self.file = [open(name, 'r') for name in self.filename]

    def read(self):

        coords = self.next()
        while coords is not None:
            yield coords
            coords = self.next()

    def next(self):
        try:
            coords = []
            for file in self.file:
                line = file.next().split()
                if self.nfiles == 1:
                    if len(line) > 2: # check if .xvg file was created using g_angle -all -ov ...
                        coords.extend(map(float, line)[2:]) # in this case all columns after the second one are important
                    else: # else the file was created using g_chi -all
                        coords.append(map(float, line)[1])
                else: # multiple input files are supposed to be only created using g_chi -all
                    coords.append(map(float, line)[1])
            return pi*np.array(coords)/180. #switch from degrees to radians before returning
        except StopIteration:
            return None


    def readlines(self):
        coords = []
        coord = self.next()
        while coord is not None:
            coords.append(coord)
            coord = self.next()
        return np.array(coords)

    def close(self):
        self.file.close()

    def __iter__(self):
        return self.read()

    readline = next