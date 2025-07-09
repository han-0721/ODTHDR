import os
import cv2
import numpy as np
import argparse
import torch
from torchvision.transforms import ToTensor
from torchvision import transforms
from tqdm import tqdm
import torchvision.transforms as transforms
from torchvision import models
from util import normalize
from skimage.metrics.simple_metrics import peak_signal_noise_ratio
import lpips

def range_compressor(x):
    return (np.log(1 + 5000 * x)) / np.log(1 + 5000)

def drago(img):
    tonemap_drago = cv2.createTonemapDrago(gamma=2.2, saturation=0.8)
    img = tonemap_drago.process(img)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *(sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
    
def calculate_ssim(img1, img2):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_lpips(img, img2, **kwargs):
    img = transform(img.astype(np.float32))
    img2 = transform(img2.astype(np.float32))

    img1_tensor = torch.unsqueeze(img, 0).cuda()
    img2_tensor = torch.unsqueeze(img2, 0).cuda()

    lpips = lpips_model(img1_tensor, img2_tensor)

    return lpips.item()
    
lpips_model = lpips.LPIPS(net='vgg').cuda()    
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def normalize(x):
    max = x.max()
    min = x.min()
    output = (x - min)/(max - min + 1e-8)
    return output