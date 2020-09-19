"""
util.py
Author: Jaewook Lee
Last edited: 20-09-15
"""


"""
internals
"""
import os
from os import path
import sys
import re
import warnings
import itertools
from subprocess import Popen
from glob import glob
from datetime import datetime
from importlib import reload
# from argparse import ArgumentParser


"""
stacks
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from sklearn.metrics import accuracy_score, jaccard_score, f1_score


"""
image processing
"""
import cv2
from skimage.io import *
from skimage.util import *
from skimage.transform import *
from skimage.color import *
from skimage import morphology
from skimage import measure
from skimage.segmentation.boundaries import find_boundaries
from scipy.ndimage.interpolation import shift
# from scipy.ndimage.morphology import * 
import imageio


"""
PyTorch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import kornia

"""
ignite
"""
# from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
# from ignite.metrics import Accuracy, Loss


"""
pretrained models
"""
import timm
import segmentation_models_pytorch as smp


"""
augmentation
"""
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


"""
extra
"""
from tqdm.auto import tqdm, trange
tqdm.pandas()

import imageio
# from sklearn.model_selection import train_test_split
# from IPython.display import Video


"""
util functions
"""
from . import config

def look(*arrays):
    """
    best inspection functionality
    """
    for arr in arrays:
        print(arr.shape, 
              arr.dtype, 
              f'[{arr.min()},\t{arr.max()}]', 
              sep='\t')


def clip(img):
    return np.clip(img, 0, 1)


def cuda(n):
    """
    return torch.device(cuda:n)
    """
    try:
        config.device = torch.device(f'cuda:{n}')
    except RuntimeError:
        config.device = torch.device('cpu')
        
    print(config.device)
    return config.device


def convert(*n):
    """
    numpy.ndarray <-> torch.Tensor converter (batch available)
    """
    if len(n) == 1:
        n = n[0]
        if isinstance(n, np.ndarray):
            if n.ndim == 4: 
                """
                (N, H, W, C) -> (N, C, H, W)
                """
                n = n.transpose(0, 3, 1, 2)
            elif n.ndim == 3: 
                """
                (H, W, C) -> (C, H, W)
                """
                n = n.transpose(2, 0, 1)
            return torch.from_numpy(n).to(config.device)
        else:
            n = n.cpu().detach().numpy()
            if n.ndim == 4: 
                """
                (N, C, H, W) -> (N, H, W, C)
                """
                n = n.transpose(0, 2, 3, 1)
            elif n.ndim == 3: 
                """
                (C, H, W) -> (H, W, C)
                """
                n = n.transpose(1, 2, 0)
            return n
    else:
        return [convert(n) for n in n]


def sqz(img):
    if isinstance(img, torch.Tensor):
        img = convert(img)
    return img.squeeze()

def small():
    plt.rcParams["figure.figsize"] = (5, 5)
    
def medium():
    plt.rcParams["figure.figsize"] = (10, 10)

def large():
    plt.rcParams["figure.figsize"] = (20, 20)

def show(*images, ncols=4, titles=None):
    """
    TODO: add usage
    
    show(np.random.rand(100, 200, 3))
    """
    n = len(images)
    
    if n == 1:
        if isinstance(images[0], pd.core.frame.DataFrame):
            df = images[0]
            return df.applymap(lambda x: x.shape if isinstance(x, np.ndarray) else x)
        elif isinstance(images[0], list):
            images = images[0]
            n = len(images)
        elif isinstance(images[0], dict):
            images = list(images[0].values())
            n = len(images)
            
        if n == 1:
            plt.figure()
            imshow(img_as_float32(sqz(images[0])))
            plt.axis('off')
            plt.show()
            return
        
    q, r = n // ncols, n % ncols
    if r > 0:
        q += 1
        
    if n < ncols:
        ncols = n
    
    H, W = sqz(images[0]).shape[:2]
    
    fig, ax = plt.subplots(q, ncols, figsize=(ncols * W/H * 4, q * 4))
    
    ax = ax.flatten()
    
    if titles:
        for idx, title in enumerate(titles):
            ax[idx].set_title(title, size=30)
        
    
    for i, img in enumerate(images):
        ax[i].imshow(img_as_float32(sqz(img)), vmin=0, vmax=1)
        ax[i].axis('off')
        
    for i in range(n, len(ax)):
        ax[i].set_visible(False)
        
        
    plt.tight_layout()
    plt.show()
    
    
def timestamp():
    return datetime.now().strftime("%m%d_%H%M%S")

def green(img):
    return np.dstack([
        img * 0, img, img * 0
    ])

# def overlay(img, mask):
#     return clip(gray2rgb(img) + green(mask)/3)

"""
Losses
"""

class ComboLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.losses = [
            nn.CrossEntropyLoss(),
            kornia.losses.DiceLoss(),
            kornia.losses.FocalLoss(
                alpha=.5, gamma=2, reduction='mean')
        ]
    
    def __call__(self, pred, true):
        return sum(loss(pred, true) 
                   for loss in self.losses)
    
    
sys.path.append('../external_packages/LovaszSoftmax/pytorch/')
sys.path.append('../../external_packages/LovaszSoftmax/pytorch/')
from lovasz_losses import lovasz_hinge, lovasz_softmax

LovaszLoss = (lambda pred, true: 
             lovasz_softmax(F.softmax(pred, dim=1), true, per_image=True))

# def read_dir(dirpath, ext="PNG"):
#     df = pd.DataFrame(dict(img_path=sorted(glob(path.join(dirpath, f"*.{ext}")))))
#     df["img"] = (df.img_path
#                  .apply(imread))
#     return df

# def list_dir(dirpath, name, ext="png"):
#     df = pd.DataFrame({name: glob(path.join(dirpath, f"*.{ext}"))})
    
#     df["ID"] = (
#         df[name]
#         .apply(lambda filepath:
#                path.splitext(path.split(filepath)[1])[0]
#         )
#     )
    
#     return df

def overlay(orange, blue):
    orange = rgb2gray(orange)
    blue = rgb2gray(blue)
    return np.dstack([
        orange,
        (orange+blue)/2,
        blue
    ])

def play(video, filepath="tmp.mp4"):
    from IPython.display import Video
    video = img_as_ubyte(clip(img_as_float32(np.asarray(video))))
    imageio.mimwrite(filepath, video, fps=24, macro_block_size=None)
    display(Video(filepath, width=800, height=400))
    
def overlay(orange, blue):
    orange = rgb2gray(orange)
    blue = rgb2gray(blue)
    return np.dstack([
        orange,
        (orange+blue)/2,
        blue
    ])

"""
image processing
"""
def largest(img):
    from skimage.measure import label
    labeled = label(img)
    largest = sorted([(region.area, region.label) for region in regionprops(labeled)], reverse=True)[0][1]
    return labeled == largest

def second_largest(img):
    from skimage.measure import label
    labeled = label(img)
    largest = sorted([(region.area, region.label) for region in regionprops(labeled)], reverse=True)[0][:2]
    return (labeled == largest[0]) | (labeled == largest[1])

"""
video reader
"""

def video_read(filename):
    return np.array(imageio.mimread(filename, memtest=False))


def figsize(w, h):
    plt.rcParams["figure.figsize"] = (w, h)

def xlarge():
    figsize(40, 40)