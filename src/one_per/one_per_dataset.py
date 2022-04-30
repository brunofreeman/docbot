import sys
from typing import Dict, List, Tuple
from enum import Enum
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import linecache


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


class ViewType(Enum):
    FRONTAL = 0
    LATERAL = 1

    def __str__(self) -> str:
        if self is ViewType.FRONTAL:
            return "frontal"
        elif self is ViewType.LATERAL:
            return "lateral"
        else:
            raise ValueError


N_TRAIN: Dict[ViewType, List[int]] = {
    ViewType.FRONTAL: [152288, 30174, 33152, 83389, 7558, 62234, 46259, 17873, 48931, 54321, 89477, 3783, 8603, 92446],
    ViewType.LATERAL: [ 25860,  8520,  7756, 11375, 2560,  6916, 11068,  4030,  6281,  8542, 17233, 1654, 2045,  8054]
}

MAX_PIXEL_INTENSITY: int = 255

COLOR_MODE: int = cv2.IMREAD_GRAYSCALE
RESOLUTION: Tuple[int, int] = (256, 256)
INTERPOLATION: int = cv2.INTER_CUBIC

DATASET_NAME: str = (
    ("greyscale" if COLOR_MODE == cv2.IMREAD_GRAYSCALE else "color") +
    f"{RESOLUTION[0]}x{RESOLUTION[1]}"
)

IMG_SRC_DIR: str = f"/groups/CS156b/2022/team_dirs/docbot/{DATASET_NAME}/train"
LABEL_CSV_PATH: str = "/groups/CS156b/data/student_labels/train.csv"
LABEL_DROP_COLUMNS: List[str] = ["Sex", "Age", "Frontal/Lateral", "AP/PA"]

PID_PREFIX: str = "pid"
STUDY_DIR: str = "/study1"
VIEW_PRIMARY: str = "/view1_frontal.jpg"
VIEW_BACKUP: str = "/view1_lateral.jpg"


def index_filename(vt: ViewType, pi: int) -> str:
    return f"./src/one_per/idx_files/{vt}_{pi:02d}.txt"


class CheXpertOnePerDataset(Dataset):
    view_type: ViewType
    pathology_i: int
    len: int
    labels: pd.DataFrame
    device: torch.device

    def __init__(self, view_type: ViewType, pathology_i: int, device: torch.device = torch.device("cpu")):
        self.view_type = view_type
        self.pathology_i = pathology_i
        self.len = N_TRAIN[self.view_type][self.pathology_i]
        self.device = device
        self.labels = pd.read_csv(LABEL_CSV_PATH)
        
        # drop unused columns
        self.labels.drop(LABEL_DROP_COLUMNS, axis=1, inplace=True)

        # drop columns for all pathologies save for our targert
        self.labels.drop([
            PATHOLOGIES[i] for i in range(len(PATHOLOGIES)) if i != self.pathology_i
        ], axis=1, inplace=True)

        # drop all rows that have NaN value for our target pathology
        self.labels.dropna(subset=[PATHOLOGIES[self.pathology_i]], inplace=True)

    def __len__(self):
        return self.len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not idx < self.len:
            raise IndexError(f"access of index {idx} in dataset of length {self.len}")

        # path to the image
        img_path: str = f"{IMG_SRC_DIR}/{self.idx_to_dir(idx)}"

        # convert the image into a PyTorch tensor and normalize [0, 255] -> [0, 1]
        img: np.ndarray = np.array([cv2.imread(img_path, COLOR_MODE)], dtype=np.float32)
        img /= MAX_PIXEL_INTENSITY
        img_tensor: torch.Tensor = torch.from_numpy(img)

        # grab the indicator vector from the label data frame
        row: pd.DataFrame = self.labels[
            self.labels["Path"].str.contains(img_path[len(IMG_SRC_DIR):])
        ]
        if len(row) != 1:
            raise ValueError(f"key of '{img_path[len(IMG_SRC_DIR):]}' yielded {len(row)} results, not 1")
        row: pd.Series = row.iloc[0]
        
        indicator: torch.Tensor = torch.FloatTensor([row[PATHOLOGIES[self.pathology_i]]])

        # return the data point as an (X, Y) pair
        return img_tensor.to(self.device), indicator.to(self.device)

    def idx_to_dir(self, idx: int) -> str:
        idx_filename: str = index_filename(self.view_type, self.pathology_i)
        line: str = linecache.getline(idx_filename, idx + 1)  # linecache uses 1-indexing
        return line[:-1]  # remove trailing newline


def main(argv: List[str]) -> None:
    dataset = CheXpertOnePerDataset(ViewType.FRONTAL, 0)
    print(len(dataset))
    print(dataset[0])
    print(dataset[len(dataset) - 1])


if __name__ == "__main__":
    main(sys.argv)
