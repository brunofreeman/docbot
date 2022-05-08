import sys
from typing import List
import numpy as np
import torch
import torch.nn as nn
import cv2
import pandas as pd
from dnop_dataset import ViewType, PATHOLOGIES, DATASET_NAME, COLOR_MODE, DNOPDataset
from dnop import SAVE_DIR, get_model, most_recent_save


CSV_DIR: str = "./out/csv"
DATASET_PATH: str = f"/groups/CS156b/2022/team_dirs/docbot/{DATASET_NAME}"
TEST_ID_PATH: str = "/groups/CS156b/data/student_labels/test_ids.csv"
ID_COL: str = "Id"


def get_csv_filepath(pi: int) -> str:
    return f"{CSV_DIR}/dnop_v1_p{pi:02d}.csv"

def one_hot_to_label(vec: torch.Tensor) -> float:
    # -1 * vec[0] + 0 * vec[1] + 1 * vec[2]
    return float(vec[2] - vec[0])

def usage(argv: List[str]) -> None:
    print(f"usage: {argv[0]} <i \u2208 [0, 13]>", flush=True)
    sys.exit(1)


def main(argv: List[str]) -> None:
    if len(argv) != 2:
        usage(argv)
    try:
        pi = int(argv[1])
    except ValueError:
        usage(argv)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    
    modelf = get_model(device)
    modelf.load_state_dict(
        torch.load(f"{SAVE_DIR}/{most_recent_save(ViewType.FRONTAL, pi)}", map_location=device)
    )
    modelf.eval()

    modell = get_model(device)
    modell.load_state_dict(
        torch.load(f"{SAVE_DIR}/{most_recent_save(ViewType.LATERAL, pi)}", map_location=device)
    )
    modell.eval()


    # use either frontal or lateral model to populate prediction for the given pathology
    def predict(id: str, path: str) -> List:
        # convert the image into a PyTorch tensor and normalize
        img: np.ndarray = np.array(cv2.imread(f"{DATASET_PATH}/{path}", COLOR_MODE), dtype=np.float32)
        img = np.tile(img, (3, 1, 1))  # convert single greyscale channel to three channels
        img_tensor: torch.Tensor = torch.from_numpy(img)
        img_tensor = DNOPDataset.NORMALIZATION(img_tensor)

        if str(ViewType.FRONTAL) in path:
            model = modelf
        elif str(ViewType.LATERAL) in path:
            model = modell
        else:
            raise ValueError

        # run model, extract the single output, run Softmax
        prediction = nn.Softmax(dim=0)(model(img_tensor[None,])[0].detach())

        return [id, one_hot_to_label(prediction)]
        
    
    # get a data frame with (Id, Path) pairs
    to_predict: pd.DataFrame = pd.read_csv(TEST_ID_PATH)

    # empty prediciton data frame
    predictions: pd.DataFrame = pd.DataFrame(columns=[ID_COL, PATHOLOGIES[pi]])

    # append a row with a prediction for each (Id, Path) test pair
    for i, row in to_predict.iterrows():
        predictions.loc[i] = predict(str(row[ID_COL]), row["Path"])
    
    # write to CSV, sorted to make combining easier
    predictions.sort_values(ID_COL, inplace=True)
    predictions.to_csv(get_csv_filepath(pi), index=False)


if __name__ == "__main__":
    main(sys.argv)
