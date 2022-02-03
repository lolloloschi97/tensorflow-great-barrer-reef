import numpy as np
import pandas as pd
import torch
import torchvision
import pickle
import json
from matplotlib import pyplot as plt
from typing import Callable, Dict, List, Tuple, Union


# path
UTILS_ROOT = "../utils/"
DATASET_ROOT = "../dataset/"
DATAFRAME_ROOT = "/dataframe/"
IMAGES_ROOT = "/images/"
LABELS_ROOT = "/labels/"
TRAIN_ROOT = "../train/"
VALIDATION_ROOT = "../validation/"

#image dimensions
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
# training parameters
BATCH_SIZE = 128
NUM_WORKER = 2
TRAIN_SIZE = 0.6
