import numpy as np
import pandas as pd
import torch
import torchvision
import pickle
import json
from matplotlib import pyplot as plt
from typing import Callable, Dict, List, Tuple, Union

WITH_COLAB = False

# path
if WITH_COLAB:
    UTILS_ROOT = "tensorflow-great-barrer-reef/utils/"
    DATASET_ROOT = "tensorflow-great-barrer-reef/dataset/"
    TRAIN_ROOT = "tensorflow-great-barrer-reef/train/"
    VALIDATION_ROOT = "tensorflow-great-barrer-reef/validation/"
    CHECKPOINT_ROOT = "tensorflow-great-barrer-reef/checkpoints"
else:
    UTILS_ROOT = "../utils/"
    DATASET_ROOT = "../dataset/"
    TRAIN_ROOT = "../train/"
    VALIDATION_ROOT = "../validation/"
    CHECKPOINT_ROOT = "../checkpoints"
DATAFRAME_ROOT = "/dataframe/"
IMAGES_ROOT = "/images/"
LABELS_ROOT = "/labels/"

#image dimensions
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
RESHAPE = False
RESHAPE_FACTOR = 4

# training parameters
EPOCHS = 10
LR = 1e-5
BATCH_SIZE = 4
NUM_WORKER = 2
TRAIN_SIZE = 0.9
if torch.cuda.is_available:
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = 'cpu'
