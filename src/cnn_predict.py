import sys
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import cv2
import pandas as pd

MODEL_PATH: str = "./out/cnn_v1_005.pt"
PREDICITON_CSV_PATH: str = "./out/cnn_v1.csv"

OUT_DIM: int = 14

MAX_PIXEL_INTENSITY: int = 255
COLOR_MODE: int = cv2.IMREAD_GRAYSCALE
RESOLUTION: Tuple[int, int] = (256, 256)
INTERPOLATION: int = cv2.INTER_CUBIC

DATASET_NAME: str = (
    ("greyscale" if COLOR_MODE == cv2.IMREAD_GRAYSCALE else "color") +
    f"{RESOLUTION[0]}x{RESOLUTION[1]}"
)

DATASET_PATH: str = f"/groups/CS156b/2022/team_dirs/docbot/{DATASET_NAME}"
TEST_ID_PATH: str = "/groups/CS156b/data/student_labels/test_ids.csv"

ID_COL: str = "Id"

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


def main(argv: List[str]) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    
    # load pre-trained model
    model = nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=(3,3)),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(p=0.1),
        
        nn.Conv2d(8, 16, kernel_size=(3,3)),
        nn.BatchNorm2d(16),
        nn.ReLU(),

        nn.Conv2d(16, 32, kernel_size=(3,3)),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Dropout(p=0.1),
        
        nn.Flatten(),
        nn.Linear(484128, 64),
        nn.ReLU(),
        nn.Linear(64, OUT_DIM)
    ).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # use model to populate predictions for a given row
    def predict(id: str, path: str) -> pd.Series:
        img_path: str = f"{DATASET_PATH}/{path}"
        img: np.ndarray = np.array([cv2.imread(img_path, COLOR_MODE)], dtype=np.float32)
        img /= MAX_PIXEL_INTENSITY
        img_tensor: torch.Tensor = torch.from_numpy(img).to(device)

        prediction = model(img_tensor[None,])[0].tolist()
        prediction.insert(0, id)
        return prediction
        
    
    # get a data frame with (Id, Path) pairs
    to_predict: pd.DataFrame = pd.read_csv(TEST_ID_PATH)

    # empty prediciton data frame
    predictions: pd.DataFrame = pd.DataFrame(columns=[ID_COL, *PATHOLOGIES])

    # append a row with a prediction for each (Id, Path) test pair
    for i, row in to_predict.iterrows():
        predictions.loc[i] = predict(str(row[ID_COL]), row["Path"])

    # write to CSV
    predictions.to_csv(PREDICITON_CSV_PATH, index=False)


if __name__ == "__main__":
    main(sys.argv)
