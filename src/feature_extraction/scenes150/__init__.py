import PIL.Image
import torch
import torchvision
import numpy as np
from loguru import logger

from src.feature_extraction.scenes150.mit_semseg.models import ModelBuilder,SegmentationModule
from src.feature_extraction.scenes150.mit_semseg.utils import colorEncode


def init_module(model_path:str=''):
    if model_path=='':
        model_path='models/scenes150'
     # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch='resnet50dilated',
        fc_dim=2048,
        weights='{}/encoder_epoch_20.pth'.format(model_path))
    net_decoder = ModelBuilder.build_decoder(
        arch='ppm_deepsup',
        fc_dim=2048,
        num_class=150,
        weights='{}/decoder_epoch_20.pth'.format(model_path),
        use_softmax=True)

    crit = torch.nn.NLLLoss(ignore_index=-1)
    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

    return segmentation_module

def extract_feature(
    segmentation_module:SegmentationModule,
    image_path:str,
    cuda:bool=False
):
    segmentation_module.eval()
    if cuda:
        segmentation_module.to(torch.device('cuda:0'))
    else:
        segmentation_module.cpu()

    # Load and normalize one image as a singleton tensor batch
    pil_to_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # These are RGB mean+std values
            std=[0.229, 0.224, 0.225])  # across a large photo dataset.
    ])

    pil_image = PIL.Image.open(image_path).convert('RGB')
    # img_original = np.array(pil_image)
    img_data = pil_to_tensor(pil_image)
    if cuda:
        singleton_batch = {'img_data': img_data[None].cuda()}
    else: 
        singleton_batch = {'img_data': img_data[None]}
    output_size = img_data.shape[1:]
    logger.info('')

    # Run the segmentation at the highest resolution.
    with torch.no_grad():
        #this line take time
        scores = segmentation_module(singleton_batch, segSize=output_size)
    
    _, pred = torch.max(scores, dim=1)

    unique_values, counts = np.unique(pred, return_counts=True)
    counts=counts.astype('float64')
    num_pixel=pred.shape[0]*pred.shape[1]*pred.shape[2]
    logger.info(pred.shape)
    # print(counts.sum())
    # print(num_pixel)
    counts/=num_pixel

    return {
        key:value for key,value in zip(unique_values,counts)
    }


