import sys
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from dataset_simple import CheXpertTrainingDataset
import os
import cv2
import pandas as pd

MODEL_PATH = "./model.pt"
N_PREDICT: int = 6454 # verified correct number of images to predict

NAN_VALUE: float = 0.0

MAX_PIXEL_INTENSITY: int = 255

COLOR_MODE: int = cv2.IMREAD_GRAYSCALE
RESOLUTION: Tuple[int, int] = (256, 256)
INTERPOLATION: int = cv2.INTER_CUBIC

DATASET_NAME: str = (
    ("greyscale" if COLOR_MODE == cv2.IMREAD_GRAYSCALE else "color") +
    f"{RESOLUTION[0]}x{RESOLUTION[1]}"
)

IMG_SRC_DIR: str = f"/groups/CS156b/2022/team_dirs/docbot/{DATASET_NAME}/solution"
# LABEL_CSV_PATH: str = f"/groups/CS156b/data/student_labels/train.csv"
# LABEL_DROP_COLUMNS: List[str] = ["Sex", "Age", "Frontal/Lateral", "AP/PA"]

PID_PREFIX: str = "pid"
STUDY_DIR: str = "/study1"
VIEW_PRIMARY: str = "/view1_frontal.jpg"
VIEW_BACKUP: str = "/view1_lateral.jpg"

PATHOLOGIES: List[str] = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices"
]

class CheXpertSolutionDataset(Dataset):
    len: int
    # labels: pd.DataFrame

    def __init__(self, device: torch.device = torch.device("cpu")):
        self.len = N_PREDICT
        self.device = device
        # self.labels = pd.read_csv(LABEL_CSV_PATH).fillna(NAN_VALUE)

        # # trim off unused data to reduce data frame size
        # self.labels.drop(LABEL_DROP_COLUMNS, axis=1, inplace=True)
        # self.labels.drop(self.labels[self.labels["Path"].str.contains(STUDY_DIR) & (
        #     self.labels["Path"].str.contains(VIEW_PRIMARY) |
        #     self.labels["Path"].str.contains(VIEW_BACKUP)
        # )].index, inplace=True)

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # get the path to the image
        img_path: str = IMG_SRC_DIR + CheXpertTrainingDataset.idx_to_dir(idx)
        if os.path.exists(img_path + VIEW_PRIMARY):
            # use the frontal shot if it exists
            img_path += VIEW_PRIMARY
        else:
            # for the extremely small amount of cases where only lateral
            # shots are available, just use that image
            img_path += VIEW_BACKUP

        if not os.path.exists(img_path):
            print(f"Dataset failed: could not find {img_path} for data point {idx}. Using all zeroes as fallback.")
            img: np.ndarray = np.zeros(shape=(1, *RESOLUTION))
        else:
            # convert the image into a PyTorch tensor and normalize [0, 255] -> [0, 1]
            img: np.ndarray = np.array([cv2.imread(img_path, COLOR_MODE)], dtype=np.float32)
            img /= MAX_PIXEL_INTENSITY
        img_tensor: torch.Tensor = torch.from_numpy(img)

        # # grab the indicator vector from the label data frame
        # row: pd.Series = self.labels[
        #     self.labels["Path"].str.contains(img_path[img_path.index(STUDY_DIR)])
        # ].iloc[0]
        # indicator: np.ndarray = torch.FloatTensor([row[p] for p in PATHOLOGIES])

        # Normally would return (X,Y) pair but we only have X 
        return img_tensor.to(self.device)

    @staticmethod
    def idx_to_dir(idx: int) -> str:
        return f"/{PID_PREFIX}{(idx + 1):05d}{STUDY_DIR}"


def main(argv: List[str]) -> None:
    dataset = CheXpertSolutionDataset()
    print(len(dataset))
    
    model = torch.load(MODEL_PATH)

    output = []

    #TODO: get code from fill average and change the lambda function to use model.output()

if __name__ == "__main__":
    main(sys.argv)
