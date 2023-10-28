from typing import Tuple,Dict,List

import numpy as np
import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
from PIL import Image
from loguru import logger

from sfivn.feature_extraction.places365 import resnet18_365
from sfivn.feature_extraction.places365 import wideresnet
import sfivn

features_blobs=[]

def hook_feature(module, input, output):
    global features_blobs
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))



def load_model(model_path:str="../models/model/places365"):
    # this model has a last conv feature map as 14x14

    model_file = '{}/wideresnet18_places365.pth.tar'.format(model_path)
    # if not os.access(model_file, os.W_OK):
        # os.system('wget http://places2.csail.mit.edu/models_places365/' + model_file)
        # os.system('wget https://raw.githubusercontent.com/csailvision/places365/master/wideresnet.py')

    
    model = wideresnet.resnet18(num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)

    
    # hacky way to deal with the upgraded batchnorm2D and avgpool layers...
    for i, (name, module) in enumerate(model._modules.items()):
        module = resnet18_365.recursion_change_bn(model)
    model.avgpool = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0)
    
    model.eval()



    # the following is deprecated, everything is migrated to python36

    ## if you encounter the UnicodeDecodeError when use python3 to load the model, add the following line will fix it. Thanks to @soravux
    #from functools import partial
    #import pickle
    #pickle.load = partial(pickle.load, encoding="latin1")
    #pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
    #model = torch.load(model_file, map_location=lambda storage, loc: storage, pickle_module=pickle)

    model.eval()
    # hook the feature extractor
    features_names = ['layer4','avgpool'] # this is the last conv layer of the resnet
    for name in features_names:
        model._modules.get(name).register_forward_hook(hook_feature)
    return model



def init_module(model_path:str,label_path:str='{}/{}'.format('/'.join(sfivn.__file__.split('/')[:-1]),'models/model_metadata/places365'))->Tuple:
        
    # load the labels
    classes, labels_IO, labels_attribute, W_attribute = resnet18_365.load_labels(label_path)

    # load the model

    
    model = load_model(model_path=model_path)

    # load the transformer
    tf = resnet18_365.returnTF() # image transformer

    # # get the softmax weight
    # params = list(model.parameters())
    # weight_softmax = params[-2].data.numpy()
    # weight_softmax[weight_softmax<0] = 0

    return classes,labels_IO,labels_attribute,W_attribute,model,tf

def extract_feature(image_path:str,classes,labels_IO,labels_attribute,W_attribute,model,tf,get_top_only:int=20)->Tuple[float,List[float],List[str]]:
    features_blobs = []
    
    img = Image.open(image_path)
    input_img = V(tf(img).unsqueeze(0))

    # forward pass
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    # logger.info('RESULT ON ' + img_url)

# output the IO prediction
    io_image = np.mean(labels_IO[idx[:10]]) # vote for the indoor or outdoor
    if io_image < 0.5:
        logger.info('--TYPE OF ENVIRONMENT: indoor')
    else:
        logger.info('--TYPE OF ENVIRONMENT: outdoor')

    # output the prediction of scene category
    logger.info('--SCENE CATEGORIES:')
    for i in range(0, 5):
        logger.info('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

    # output the scene attributes
    # responses_attribute = W_attribute.dot(features_blobs[1])
    # idx_a = np.argsort(responses_attribute)
    # logger.info('--SCENE ATTRIBUTES:')
    # scences_attributes=[labels_attribute[idx_a[i]] for i in range(-1,-10,-1)]
    # logger.info(', '.join(scences_attributes))



    return (
        io_image,
        {
            key:value for key,value in zip(idx[:get_top_only],probs[:get_top_only])
        },
    )
  
  
def extract_feature_full(image_path:str,classes,labels_IO,labels_attribute,W_attribute,model,tf,get_top_only:int=20)->Tuple[float,List[float],List[str]]:
    features_blobs = []
    
    img = Image.open(image_path)
    input_img = V(tf(img).unsqueeze(0))

    # forward pass
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    # logger.info('RESULT ON ' + img_url)

# output the IO prediction
    io_image = np.mean(labels_IO[idx[:10]]) # vote for the indoor or outdoor
    if io_image < 0.5:
        logger.info('--TYPE OF ENVIRONMENT: indoor')
    else:
        logger.info('--TYPE OF ENVIRONMENT: outdoor')

    # output the prediction of scene category
    logger.info('--SCENE CATEGORIES:')
    for i in range(0, 5):
        logger.info('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

    # output the scene attributes
    # responses_attribute = W_attribute.dot(features_blobs[1])
    # idx_a = np.argsort(responses_attribute)
    # logger.info('--SCENE ATTRIBUTES:')
    # scences_attributes=[labels_attribute[idx_a[i]] for i in range(-1,-10,-1)]
    # logger.info(', '.join(scences_attributes))



    return (
        io_image,
        probs.tolist()
    )
    
# classes, labels_IO, labels_attribute, W_attribute = load_labels()

# # load the model

# features_blobs = []
# model = load_model()

# # load the transformer
# tf = returnTF() # image transformer

# # get the softmax weight
# params = list(model.parameters())
# weight_softmax = params[-2].data.numpy()
# weight_softmax[weight_softmax<0] = 0

# # load the test image
# # img_url = 'http://places.csail.mit.edu/demo/6.jpg'
# # os.system('wget %s -q -O test.jpg' % img_url)
# img = Image.open('../data/images/7JJfJgyHYwU/frame_24.jpg')
# input_img = V(tf(img).unsqueeze(0))

# # forward pass
# logit = model.forward(input_img)
# h_x = F.softmax(logit, 1).data.squeeze()
# probs, idx = h_x.sort(0, True)
# probs = probs.numpy()
# idx = idx.numpy()

# # logger.info('RESULT ON ' + img_url)

# # output the IO prediction
# io_image = np.mean(labels_IO[idx[:10]]) # vote for the indoor or outdoor
# if io_image < 0.5:
#     logger.info('--TYPE OF ENVIRONMENT: indoor')
# else:
#     logger.info('--TYPE OF ENVIRONMENT: outdoor')

# # output the prediction of scene category
# logger.info('--SCENE CATEGORIES:')
# for i in range(0, 5):
#     logger.info('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))  
# classes, labels_IO, labels_attribute, W_attribute = load_labels()

# # load the model

# features_blobs = []
# model = load_model()

# # load the transformer
# tf = returnTF() # image transformer

# # get the softmax weight
# params = list(model.parameters())
# weight_softmax = params[-2].data.numpy()
# weight_softmax[weight_softmax<0] = 0

# # load the test image
# # img_url = 'http://places.csail.mit.edu/demo/6.jpg'
# # os.system('wget %s -q -O test.jpg' % img_url)
# img = Image.open('../data/images/7JJfJgyHYwU/frame_24.jpg')
# input_img = V(tf(img).unsqueeze(0))

# # forward pass
# logit = model.forward(input_img)
# h_x = F.softmax(logit, 1).data.squeeze()
# probs, idx = h_x.sort(0, True)
# probs = probs.numpy()
# idx = idx.numpy()

# # logger.info('RESULT ON ' + img_url)

# # output the IO prediction
# io_image = np.mean(labels_IO[idx[:10]]) # vote for the indoor or outdoor
# if io_image < 0.5:
#     logger.info('--TYPE OF ENVIRONMENT: indoor')
# else:
#     logger.info('--TYPE OF ENVIRONMENT: outdoor')

# # output the prediction of scene category
# logger.info('--SCENE CATEGORIES:')
# for i in range(0, 5):
#     logger.info('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))