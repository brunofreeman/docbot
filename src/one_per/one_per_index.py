import sys
import os
from typing import List
import numpy as np
import pandas as pd
from one_per_dataset import PATHOLOGIES, ViewType, N_TRAIN, index_filename

TRAIN_PATH: str = "/groups/CS156b/data/train"
LABEL_PATH: str = "/groups/CS156b/data/student_labels/train.csv"


def path_push(path: str, dir: str) -> str:
    return f"{path}/{dir}"


def path_pop(path: str) -> str:
    return path[:path.rindex("/")]


def non_nan(labels: pd.DataFrame, dir: str, pi: int) -> bool:
    row: pd.DataFrame = labels[labels["Path"].str.contains(dir)]
    assert(len(row) == 1)
    row: pd.Series = row.iloc[0]
    return not np.isnan(row[PATHOLOGIES[pi]])


def usage(argv: List[str]) -> None:
    print(f"usage: {argv[0]} {{f, l}} <i \u2208 [0, 13]>")
    sys.exit(1)


def main(argv: List[str]) -> None:
    if len(argv) != 3:
        usage(argv)
    if argv[1] == 'f':
        view_type = ViewType.FRONTAL
    elif argv[1] == 'l':
        view_type = ViewType.LATERAL
    else:
        usage(argv)
    
    try:
        pi = int(argv[2])
        if not (0 <= pi < len(PATHOLOGIES)):
            usage(argv)
    except ValueError:
        usage(argv)

    labels: pd.DataFrame = pd.read_csv(LABEL_PATH)
    f = open(index_filename(view_type, pi), "w")

    n_write: int = 0
    path: str = TRAIN_PATH

    for pid_dir in os.listdir(path):
        path = path_push(path, pid_dir)

        for study_dir in os.listdir(path):
            path = path_push(path, study_dir)

            for img_name in os.listdir(path):
                dataset_dir: str = f"{pid_dir}/{study_dir}/{img_name}"
                if str(view_type) in dataset_dir and non_nan(labels, dataset_dir, pi):
                    f.write(dataset_dir + "\n")
                    n_write += 1

            path = path_pop(path)
        path = path_pop(path)

    f.close()
    print(f"Expected lines: {N_TRAIN[view_type][pi]}, actual lines: {n_write}")


if __name__ == "__main__":
    main(sys.argv)
