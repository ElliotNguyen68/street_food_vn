from curses import meta
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import numpy as np
from PIL import Image
import cv2


 # hacky way to deal with the Pytorch 1.0 update
def recursion_change_bn(module):
    if isinstance(module, torch.nn.BatchNorm2d):
        module.track_running_stats = 1
    else:
        for i, (name, module1) in enumerate(module._modules.items()):
            module1 = recursion_change_bn(module1)
    return module

def load_labels(metadata_365_path:str='../models/model_metadata/places365'):
    # prepare all the labels
    # scene category relevant
    file_name_category = '{}/categories_places365.txt'.format(metadata_365_path)
    # if not os.access(file_name_category, os.W_OK):
    #     synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
    #     os.system('wget ' + synset_url)
    classes = list()
    with open(file_name_category) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # indoor and outdoor relevant
    file_name_IO = '{}/IO_places365.txt'.format(metadata_365_path)
    # if not os.access(file_name_IO, os.W_OK):
    #     synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt'
    #     os.system('wget ' + synset_url)
    with open(file_name_IO) as f:
        lines = f.readlines()
        labels_IO = []
        for line in lines:
            items = line.rstrip().split()
            labels_IO.append(int(items[-1]) -1) # 0 is indoor, 1 is outdoor
    labels_IO = np.array(labels_IO)

    # scene attribute relevant
    file_name_attribute = '{}/labels_sunattribute.txt'.format(metadata_365_path)
    # if not os.access(file_name_attribute, os.W_OK):
    #     synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/labels_sunattribute.txt'
    #     os.system('wget ' + synset_url)
    with open(file_name_attribute) as f:
        lines = f.readlines()
        labels_attribute = [item.rstrip() for item in lines]
    file_name_W = '{}/W_sceneattribute_wideresnet18.npy'.format(metadata_365_path)
    # if not os.access(file_name_W, os.W_OK):
    #     synset_url = 'http://places2.csail.mit.edu/models_places365/W_sceneattribute_wideresnet18.npy'
    #     os.system('wget ' + synset_url)
    W_attribute = np.load(file_name_W)

    return classes, labels_IO, labels_attribute, W_attribute


def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def returnTF():
# load the image transformer
    tf = trn.Compose([
        trn.Resize((224,224)),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tf
