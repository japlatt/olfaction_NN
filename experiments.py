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