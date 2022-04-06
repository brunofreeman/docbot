import sys
import os
from typing import List
import pandas as pd
import numpy as np

LABEL_CSV_FILEPATH: str = \
    f"{os.path.abspath(os.sep)}groups/CS156b/data/student_labels/train.csv"


def main(argv: List[str]) -> None:
    df: pd.DataFrame = pd.read_csv(LABEL_CSV_FILEPATH)
    print(df.columns)
    A: np.ndarray = np.random.uniform(0, 1, (5, 5))
    print(A)


if __name__ == "__main__":
    main(sys.argv)
