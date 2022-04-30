import sys
from typing import Dict, List
import pandas as pd


TRAIN_LABEL_CSV_FILEPATH: str = "/groups/CS156b/data/student_labels/train.csv"
TEST_ID_CSV_FILE_PATH: str    = "/groups/CS156b/data/student_labels/test_ids.csv"

PATHOLOGIES: List[str] = (
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
)

NAN_IGNORE_OUTPUT_CSV: str = "./out/csv/avg_nan_ignore.csv"
NAN_MINUS1_OUTPUT_CSV: str = "./out/csv/avg_nan_minus1.csv"


def main(argv: List[str]) -> None:
    df_labels: pd.DataFrame = pd.read_csv(TRAIN_LABEL_CSV_FILEPATH)
    avg: Dict[str, float] = dict()

    # predict with NaN ignored
    for pathology in PATHOLOGIES:
        # .mean() ignores NaN entries
        avg[pathology] = df_labels[pathology].mean()
    
    df_predict: pd.DataFrame = pd.read_csv(TEST_ID_CSV_FILE_PATH).drop("Path", axis=1).rename(columns={"Id": "ID"})
    for pathology in PATHOLOGIES:
        df_predict[pathology] = df_predict.apply(lambda _ : avg[pathology], axis=1)
    
    df_predict.to_csv(NAN_IGNORE_OUTPUT_CSV, index=False)

    # predict with Nan = -1 (not present)
    df_labels = df_labels.fillna(-1)

    for pathology in PATHOLOGIES:
        avg[pathology] = df_labels[pathology].mean()
    
    for pathology in PATHOLOGIES:
        df_predict[pathology] = df_predict.apply(lambda _ : avg[pathology], axis=1)
    
    df_predict.to_csv(NAN_MINUS1_OUTPUT_CSV, index=False)


if __name__ == "__main__":
    main(sys.argv)
