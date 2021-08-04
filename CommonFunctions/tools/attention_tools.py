from matplotlib import pyplot as plt
import numpy as np
from IPython import display
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torch.utils import data
from torchvision import transforms
import time
import hashlib
import os
import tarfile
import zipfile
import requests
import sys
sys.path.append('./')
import common_tools
import sequence_tools
import collections
import re
import random
import math


__all__ = ['show_heatmaps']


#@save
def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5), cmap='Reds'):
    """Show heatmaps of matrices."""
    common_tools.use_svg_display()
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = common_tools.plt.subplots(num_rows, 
                                          num_cols, 
                                          figsize=figsize, 
                                          sharex=True, 
                                          sharey=True, 
                                          squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6)

