import sys
import os
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import cv2
import pandas as pd
from one_per_dataset import ViewType, PATHOLOGIES
from one_per import OUT_DIR, get_model, get_save_filepath, \
    get_save_filename, is_save_filename, extract_epoch


PREDICITON_CSV_PATH: str = "./out/csv/one_per_v2.csv"

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


def most_recent_save_filepath(vt: ViewType, pi: int) -> str:
    e: int = 0
    for pt_file in os.listdir(OUT_DIR):
        if is_save_filename(pt_file):
            epoch: int = extract_epoch(pt_file)
            if get_save_filename(vt, pi, epoch) == pt_file:
                e = max(e, epoch)
    if e == 0:
        raise FileNotFoundError(f"no model to load for {{view_type: {vt}, pathology: {pi:02d}}}")
    else:
        return get_save_filepath(vt, pi, e)


def main(argv: List[str]) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    
    models: Dict[ViewType, List[nn.Sequential]] = dict()
    
    for vt in (ViewType.FRONTAL, ViewType.LATERAL):
        models[vt] = [get_model(device) for _ in range(len(PATHOLOGIES))]

    for vt in (ViewType.FRONTAL, ViewType.LATERAL):
        for pi, model in enumerate(models[vt]):
            save_filepath: str = most_recent_save_filepath(vt, pi)
            model.load_state_dict(torch.load(save_filepath, map_location=device))
            model.eval()
            print(f"loaded {save_filepath}")

    # use models to populate predictions for a given row
    def predict(id: str, path: str) -> pd.Series:
        img_path: str = f"{DATASET_PATH}/{path}"
        img: np.ndarray = np.array([cv2.imread(img_path, COLOR_MODE)], dtype=np.float32)
        img /= MAX_PIXEL_INTENSITY
        img_tensor: torch.Tensor = torch.from_numpy(img).to(device)

        prediction = [0] * len(PATHOLOGIES)

        if str(ViewType.FRONTAL) in path:
            vt = ViewType.FRONTAL
        elif str(ViewType.LATERAL) in path:
            vt = ViewType.LATERAL
        else:
            raise ValueError
        
        for pi, model in enumerate(models[vt]):
            predictions[pi] = model(img_tensor[None,])[0].detach().numpy()[0]

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
