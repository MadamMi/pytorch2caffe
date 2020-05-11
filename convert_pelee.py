import torch
from torch.autograd import Variable
from model.configs.CC import Config
from model.configs.pelee_snet_138 import PeleeNet, load_model
import pytorch_to_caffe
# from efficientnet_pytorch import EfficientNet
# from model.resnext_101_32x4d import resnext_101_32x4d
# from model.resnext_101_64x4d import resnext_101_64x4d
import cv2
import numpy as np


model = load_model(pretrained_model_path='/home/khy/arithmetic/PytorchToCaffe/weights/testmodel_best.pth.tar',
                   model_classes=138, data_classes=138)
model.eval()
input_var = Variable(torch.rand(1, 3, 224, 224))
pytorch_to_caffe.trans_net(model, input_var, 'peleenet_scene')
pytorch_to_caffe.save_prototxt('/home/khy/arithmetic/PytorchToCaffe/model/pelee_scene_138.prototxt')
pytorch_to_caffe.save_caffemodel('/home/khy/arithmetic/PytorchToCaffe/model/pelee_scene_138.caffemodel')

"""target_platform = "proxyless_cpu"
model = torch.hub.load("mit-han-lab/ProxylessNAS", target_platform, pretrained=True)
model.eval()

input_var = Variable(torch.rand(1, 3, 224, 224))
pytorch_to_caffe.trans_net(model, input_var, 'proxyless')
pytorch_to_caffe.save_prototxt('/home/khy/PycharmProjects/torch2caffetest/model/proxyless.prototxt')
pytorch_to_caffe.save_caffemodel('/home/khy/PycharmProjects/torch2caffetest/model/proxyless.caffemodel')"""

