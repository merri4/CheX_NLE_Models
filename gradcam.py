# !pip install pytorch-gradcam
# !pip install grad-cam
# !pip install timm

# ===========================================================
# DEP
# ===========================================================

import warnings
warnings.filterwarnings('ignore')
from torchvision import models
import numpy as np
import cv2
import requests
from gradcam import gradcam

import torch
import torch.nn as nn


from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
from PIL import Image
from torchvision.models import resnet50
from torchvision.models import densenet121
from torchvision.models import vit_b_16
from torchvision.transforms import functional as F


import timm.models.vision_transformer
from functools import partial

import matplotlib.pyplot as plt
import json as js


# ===========================================================
# ViT class
# ===========================================================

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome

def vit_small_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension, like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def read_jsonl_lines(input_path):
    with open(input_path) as f:
        lines = f.readlines()
        return [js.loads(l.strip()) for l in lines]


# ===========================================================
# model loading
# ===========================================================


num_classes = 14  # Set the desired number of classes


### DenseNet
model_densenet = densenet121(pretrained=True)

num_ftrs = model_densenet.classifier.in_features
model_densenet.classifier = torch.nn.Linear(num_ftrs, num_classes)
model_densenet.load_state_dict(torch.load("/content/drive/MyDrive/MIMIC_Dataset/weights/densenet121_CXR_0.3M_mae.pth")['model'])
model_densenet.cuda()
model_densenet.eval()
target_layers_densenet = [model_densenet.features.denseblock4]



### DenseNet-mocov2
model_densenet_mocov2 = densenet121(pretrained=True)

num_ftrs = model_densenet_mocov2.classifier.in_features
model_densenet_mocov2.classifier = torch.nn.Linear(num_ftrs, num_classes)
model_densenet_mocov2.load_state_dict(torch.load("/content/drive/MyDrive/MIMIC_Dataset/weights/densenet121_CXR_0.3M_mocov2.pth")['model'])
model_densenet_mocov2.cuda()
model_densenet_mocov2.eval()
target_layers_densenet_mocov2 = [model_densenet_mocov2.features.denseblock4]



### Resnet
model_resnet = resnet50(pretrained=True)

num_ftrs = model_resnet.fc.in_features
model_resnet.fc = torch.nn.Linear(num_ftrs, num_classes)
model_resnet.load_state_dict(torch.load("/content/drive/MyDrive/MIMIC_Dataset/weights/resnet50_imagenet_swav.pth")['model'])
model_resnet.cuda()
model_resnet.eval()
target_layers_resnet = [model_resnet.layer4[-1]]



### ViT
model_vit = vit_small_patch16(num_classes=14, drop_path_rate=0.2, global_pool=True)
checkpoint = torch.load('/content/drive/MyDrive/MIMIC_Dataset/weights/vit-s_CXR_0.3M_mae.pth', map_location='cpu')
model_vit.load_state_dict(checkpoint['model'], strict=True)
model_vit.cuda()
model_vit.eval()
target_layers_vit = [model_vit.blocks[-1].norm1]



# ===========================================================
# Param Setting
# ===========================================================

mean = [0.5056, 0.5056, 0.5056]
std = [0.252, 0.252, 0.252]

class_name = [
    'Atelectasis',
    'Cardiomegaly',
    'Effusion',
    'Infiltrate',
    'Mass',
    'Nodule',
    'Pneumonia',
    'Pneumothorax',
    'Consolidation',
    'Edema',
    'Emphysema',
    'Fibrosis',
    'Pleural_Thickening',
    'Hernia'
]

all_targets = read_jsonl_lines("result/grad_targets.json")

thres = 4
base_dir = "./files"


# ===========================================================
# Run
# ===========================================================
for target in all_targets :
    
    torch.cuda.empty_cache()
    print("now generating patient {}...".format(target["id"]))

    # /files/s50014127
    image_dir = base_dir + "/" + target["id"]
    
    for image in target["image_name"] :
        
        # /files/s50014127/73da0836-553a87de-58ef0562-f9c31de6-c47927ac
        image_path = image_dir + image + ".jpg"

        # /files/s50014127/73da0836-553a87de-58ef0562-f9c31de6-c47927ac_Atelectasis_gradcam.png
        for class_idx, pathology_name in enumerate(class_name) :

            targets = [ClassifierOutputTarget(class_idx)]

            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
            img = np.float32(img) / 255
            input_tensor = preprocess_image(img, mean=mean, std=std)

            f, axarr = plt.subplots(1,4, figsize=(15, 10))
            plt.subplots_adjust(wspace=0.5)
            axarr[0].title.set_text('DenseNet-121-MAE')
            axarr[1].title.set_text('DenseNet-121-mocov2')
            axarr[2].title.set_text('ResNet-50')
            axarr[3].title.set_text('ViT')


            with GradCAM(model=model_densenet, target_layers=target_layers_densenet, use_cuda=True) as cam:
                grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
                cam_img_densenet = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)
                axarr[0].imshow(cam_img_densenet)

            with GradCAM(model=model_densenet_mocov2, target_layers=target_layers_densenet_mocov2, use_cuda=True) as cam:
                grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
                cam_img_densenet_mocov2 = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)
                axarr[1].imshow(cam_img_densenet_mocov2)

            with GradCAM(model=model_resnet, target_layers=target_layers_resnet, use_cuda=True) as cam:
                grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
                cam_img_resnet = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)    
                axarr[2].imshow(cam_img_resnet)

            with GradCAM(model=model_vit, target_layers=target_layers_vit, use_cuda=True) as cam:
                grayscale_cams = cam(input_tensor=input_tensor, targets=targets)
                cam_img_vit = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)
                plt.imshow(cam_img_vit)
                axarr[3].imshow(cam_img_vit)

            save_path = image_path[:-4] + "_" + pathology_name + "_gradcam.png"
            plt.savefig(save_path)
            plt.close()