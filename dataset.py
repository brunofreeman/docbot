import os
os.environ['OPENBLAS_NUM_THREADS'] = '8'
import sys
from turtle import up
from typing import Dict, FrozenSet, List, Set, Tuple
import pandas as pd
from tabulate import tabulate


TRAIN_LABEL_CSV_FILEPATH: str = "/groups/CS156b/data/student_labels/train.csv"
TEST_ID_CSV_FILE_PATH: str    = "/groups/CS156b/data/student_labels/test_ids.csv"

PATHOLOGIES: Tuple[str] = (
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


def main(argv: List[str]) -> None:
    df_labels: pd.DataFrame = pd.read_csv(TRAIN_LABEL_CSV_FILEPATH)

    avg_nan: float = df_labels.apply(lambda r : r.isnull().sum(), axis=1).mean()
    print(f"average missing data points: {avg_nan}")

    study_to_views: Dict[str, Tuple[int, int]] = dict()
    patient_counts: Dict[str, int] = dict()
    outlier_files: List[str] = list()

    def update_dict(path: str) -> None:
        if (
            not (path.startswith("train/") and "unexpected path head") or
            not (path.endswith("_frontal.jpg") or path.endswith("_lateral.jpg") and f"unexpected path tail")
        ):
            outlier_files.append(path)
            return

        study_id: str = path[:path.rindex("/")]
        patient_id: str = study_id[study_id.index("/") + 1:-1]
        assert(patient_id.startswith("pid"))
        filename: str = path[path.rindex("/") + 1 : path.rindex(".")]
        assert(filename.startswith("view"))
        is_frontal: bool = "frontal" in filename

        if patient_id not in patient_counts:
            patient_counts[patient_id] = 0
        patient_counts[patient_id] += 1
        
        if study_id not in study_to_views:
            study_to_views[study_id] = (0, 0)
        
        study_to_views[study_id] = (
            study_to_views[study_id][0] + is_frontal,
            study_to_views[study_id][1] + (not is_frontal)
        )

    df_labels["Path"].apply(update_dict)

    view_counts: Dict[Tuple[int, int], int] = dict()
    for views in study_to_views.values():
        if views not in view_counts:
            view_counts[views] = 0
        view_counts[views] += 1
    
    print(tabulate(
        sorted(view_counts.items(), key=lambda x : x[1], reverse=True),
        headers=["View Config", "Study Count"],
        tablefmt="fancy_grid"
    ))

    patient_ag_counts: Dict[int, int] = dict()
    for cnt in patient_counts.values():
        if cnt not in patient_ag_counts:
            patient_ag_counts[cnt] = 0
        patient_ag_counts[cnt] += 1
    
    print(tabulate(
        sorted(patient_ag_counts.items(), key=lambda x : x[0]),
        headers=["Study Count", "No. Patients w/ Study Count"],
        tablefmt="fancy_grid"
    ))

    print(f"outlier files not analyzed: {outlier_files}")




if __name__ == "__main__":
    main(sys.argv)
