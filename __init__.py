'''
These imports are shared across all files.
Created by Basile Van Hoorick.
'''

# Library imports.
import argparse
import collections
import collections.abc
import copy
import cv2
import imageio
import json
import logging
import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pathlib
import pickle
import platform
import random
import scipy
import seaborn as sns
import shutil
import sklearn
import sys
import time
import torch
import torch.nn
import torch.nn.functional
import torch.optim
import torch.utils
import torch.utils.data
import torchvision
import torchvision.datasets
import torchvision.models
import torchvision.transforms
import torchvision.utils
import torch_cluster
import tqdm
import wandb
import warnings
from einops import rearrange, repeat

PROJECT_NAME = 'occlusions-4d'

sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'data/'))
sys.path.append(os.path.join(os.getcwd(), 'eval/'))
sys.path.append(os.path.join(os.getcwd(), 'model/'))
sys.path.append(os.path.join(os.getcwd(), 'utils/'))
