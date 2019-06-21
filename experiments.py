import numpy as np
import os.path
from struct import unpack
import sys
if sys.version_info[0] < 3:
    import cPickle as pickle
else:
    import pickle


'''
Random constant current input
N: size of array
p: probability of index having input
I_inp: strength of current
'''
def get_rand_I(N, p, I_inp):
    I = np.random.random(N)
    I[I > p] = 0
    I[np.nonzero(I)] = I_inp
    return I

'''
A time dependent function - will need to use a network_operation in training and test script
@network_operation(dt)
def f(t):
    stuff you want it to do
tr
    rise time, include time units (i.e. 1*ms)
    must be defined in run script
tf
    fall time, include time units
    must be defined in run script
width
    how long the current should be at max input
    must be defined in run script
max_inp
    max input, include units
    must be defined in runscript
    can just provide units if intensity of current is set by the active variable
'''
def get_gradual_current():
    I = '2*(0.5*(1-tanh(-3.5*(t-tstart)/tr)) - 0.5)*0.5*(1-tanh(-3.5*(width+tr-(t-tstart))/tf))*max_inp'
    return I



'''
MNIST helper function to load and unpack MNIST data.  To run you need to download
the MNIST dataset from http://yann.lecun.com/exdb/mnist/.

picklename: name of file to write/read data from
MNIST_data_path: path to the MNIST data folder
bTrain: if tr
'''
def get_labeled_data(picklename, MNIST_data_path, bTrain = True):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """
    if os.path.isfile('%s.pickle' % picklename):
        data = pickle.load(open('%s.pickle' % picklename, 'rb'))#, encoding='utf-8')
    else:
        # Open the images with gzip in read binary mode
        if bTrain:
            images = open(MNIST_data_path + 'train-images.idx3-ubyte','rb')
            labels = open(MNIST_data_path + 'train-labels.idx1-ubyte','rb')
        else:
            images = open(MNIST_data_path + 't10k-images.idx3-ubyte','rb')
            labels = open(MNIST_data_path + 't10k-labels.idx1-ubyte','rb')
        # Get metadata for images
        images.read(4)  # skip the magic_number
        number_of_images = unpack('>I', images.read(4))[0]
        rows = unpack('>I', images.read(4))[0]
        cols = unpack('>I', images.read(4))[0]
        # Get metadata for labels
        labels.read(4)  # skip the magic_number
        N = unpack('>I', labels.read(4))[0]

        if number_of_images != N:
            raise Exception('number of labels did not match the number of images')
        # Get the data
        x = np.zeros((N, rows, cols), dtype=np.uint8)  # Initialize numpy array
        y = np.zeros((N, 1), dtype=np.uint8)  # Initialize numpy array
        for i in range(N):
            if i % 1000 == 0:
                print("i: %i" % i)
            x[i] = [[unpack('>B', images.read(1))[0] for unused_col in range(cols)]  for unused_row in range(rows) ]
            y[i] = unpack('>B', labels.read(1))[0]

        data = {'x': x, 'y': y, 'rows': rows, 'cols': cols}
        pickle.dump(data, open("%s.pickle" % picklename, "wb"), -1)
    return data
'''
Takes name of file containing input stimulus and converts it to timedarray to
be used as input. Needs filename, dt with units, and the unit of the stimulus.

NOT TESTED
'''
def recorded_input(filename,dt,unit):
    if filename[-3:] is 'npy':
        stim = np.load(filename)
    else:
        stim = np.loadtxt(filename)
    stim = TimedArray(stim*unit,dt=dt)
    return stim
