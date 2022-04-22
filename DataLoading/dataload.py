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
import os

# trn_data = TensorDataset(
#     Tensor(trn_x),
#     Tensor(trn_y).type(torch.LongTensor)
# )


# tst_data = TensorDataset(
#     Tensor(tst_x),
#     Tensor(tst_y).type(torch.LongTensor)
# )

X_data = []
# files = glob.glob("../../greyscale256x256/train/**/**/*.jpg")

directories = ("train", "test", "solution")
COLOR_MODE = cv2.IMREAD_GRAYSCALE
RESOLUTION  = (256, 256)
INTERPOLATION  = cv2.INTER_CUBIC

i = 0
for category in directories:
    # path = "../../greyscale256x256/" + category
    cat_path = "../../greyscale256x256/" + category
    for pid_dir in os.listdir(cat_path):
            if i > 0:
                exit()
            i += 1
            pid_path = cat_path + "/"+  pid_dir
            img_path = pid_path + "/study1/view1_frontal.jpg"
            img = cv2.imread(img_path, COLOR_MODE)
            
            pid = pid_dir[3:9]
            datapoint = Tensor(np.array([int(pid), img]))
            print(datapoint)
            
            X_data.append(datapoint)

            # TODO: should we and how to store pid in the tensor? Tensor doesnt allow jagged shape
            # TODO handle the cases where the only study is not frontal -> guess to fill all zeros in interest of time

