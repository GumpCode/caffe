
# coding: utf-8

# # Net Surgery
#
# Caffe networks can be transformed to your particular needs by editing the model parameters. The data, diffs, and parameters of a net are all exposed in pycaffe.
#
# Roll up your sleeves for net surgery with pycaffe!

import numpy as np

# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# ## Designer Filters
#
# To show how to load, manipulate, and save parameters we'll design our own filters into a simple network that's only a single convolution layer. This net has two blobs, `data` for the input and `conv` for the convolution output and one parameter `conv` for the convolution filter weights and biases.

# Load the net, list its data and params, and filter an example image.
train_bn_prototxt = 'train_bn.prototxt'
train_prototxt = 'train.prototxt'
train_bn_caffemodel = 'VGG_coco_SSD_300x300_iter_1.caffemodel'
train_caffemodel = 'VGG_coco_SSD_300x300_iter_30000.caffemodel'
caffe.set_mode_cpu()
net = caffe.Net(train_prototxt, train_caffemodel, caffe.TEST)
bn_net = caffe.Net(train_bn_prototxt, train_bn_caffemodel, caffe.TEST)

#get the data from net
print net.params.keys()
keys = net.params.keys()
keys.remove('conv4_3_norm')
print keys
for key in keys:
    weight = net.params[key][0].data
    #bias = net.params[key][1].data
    print net.params[key][1].data.shape
    print key

params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in keys}

#get the data from net_bn
bn_params = {pr_bn: (bn_net.params[pr_bn][0].data, bn_net.params[pr_bn][1].data) for pr_bn in keys}

#send the data in net to net_bn
for key in keys:
    bn_params[key][0].flat = params[key][0].flat
    bn_params[key][1][...] = params[key][1]


bn_net.save('ssd_bn_300x300.caffemodel')
