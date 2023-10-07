import os

import torch
from PIL import Image
from torchvision import transforms
from loguru import logger

import sfivn


def init_module(model_path: str, torchvision_repo_dir: str = None):
    if torchvision_repo_dir == None:
        torchvision_repo_dir = "{}/{}".format(
            "/".join(sfivn.__file__.split("/")[:-1]), "models/pytorch_vision"
        )
    logger.info(torchvision_repo_dir)
    model = torch.hub.load(
        source="local", repo_or_dir=torchvision_repo_dir, model="inception_v3"
    )

    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def extract_feature(
    model: torch.nn.Module, image_path: str, cuda: bool = False, get_top_only: int = 20
):
    input_image = Image.open(image_path)
    preprocess = transforms.Compose(
        [
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(
        0
    )  # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    # if torch.cuda.is_available():
    #     input_batch = input_batch.to('cuda')
    #     model.to('cuda')
    model.to("cpu")
    with torch.no_grad():
        output = model(input_batch)
    # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    probs, idx = probabilities.sort(0, True)
    logger.info("{}, {}".format(probs[:5].tolist()), idx[:5].tolist())
    return {
        key.item(): value.item()
        for key, value in zip(idx[:get_top_only], probs[:get_top_only])
    }
