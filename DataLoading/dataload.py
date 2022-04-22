import torch
import torchvision
import torchvision.transforms as transforms

from PIL import Image

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader, TensorDataset
import pickle
import glob
import cv2


# trn_data = TensorDataset(
#     Tensor(trn_x),
#     Tensor(trn_y).type(torch.LongTensor)
# )


# tst_data = TensorDataset(
#     Tensor(tst_x),
#     Tensor(tst_y).type(torch.LongTensor)
# )

X_data = []
files = glob.glob("../../greyscale256x256/train/**/**/*.jpg")

# for file in files:
    