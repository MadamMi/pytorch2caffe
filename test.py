import os,sys
caffe_root = '/home/khy/use_lib/SSD/caffe/'
sys.path.insert(0, caffe_root+'python')
import caffe

model_def = '/home/khy/arithmetic/PytorchToCaffe/Pelee.prototxt'
model_weights = '/home/khy/arithmetic/PytorchToCaffe/Pelee.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)