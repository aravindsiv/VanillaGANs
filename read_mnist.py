'''
Heavily adapted from: http://g.sweyla.com/blog/2012/mnist-numpy/
'''
from __future__ import division
import os, sys, struct
from array import array as pyarray
import numpy as np
import cPickle as pickle

path = "."

image_fname = os.path.join(path, 'train-images-idx3-ubyte')
label_fname = os.path.join(path, 'train-labels-idx1-ubyte')

# if sys.argv[1] == "train":
# 	image_fname = os.path.join(path, 'train-images-idx3-ubyte')
# 	label_fname = os.path.join(path, 'train-labels-idx1-ubyte')
# elif sys.argv[1] == "test":
# 	image_fname = os.path.join(path, 't10k-images-idx3-ubyte')
# 	label_fname = os.path.join(path, 't10k-labels-idx1-ubyte')
# else:
# 	raise ValueError("Invalid argument")

label_file = open(label_fname, 'rb')
magic_nr, size = struct.unpack(">II",label_file.read(8))
lbls = pyarray("b",label_file.read())
label_file.close()

image_file = open(image_fname, 'rb')
magic_nr, size, rows, cols = struct.unpack(">IIII",image_file.read(16))
imgs = pyarray("B",image_file.read())
image_file.close()

indices = [k for k in range(size) if lbls[k] in np.arange(10)]
N = len(indices)

images = np.zeros((N,rows,cols), dtype=np.uint8)
labels = np.zeros((N,1), dtype=np.int8)

for i in range(N):
	images[i] = np.array(imgs[indices[i]*rows*cols:(indices[i]+1)*rows*cols]).reshape((rows,cols))
	labels[i] = lbls[indices[i]]

with open("train_data.pkl",'w') as f:
	pickle.dump([images,labels],f)