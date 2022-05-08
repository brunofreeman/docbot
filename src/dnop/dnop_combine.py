import sys
from typing import List
import pandas as pd
from dnop_dataset import PATHOLOGIES
from dnop_predict import get_csv_filepath, CSV_DIR, ID_COL


COMBINED_CSV_PATH: str = f"{CSV_DIR}/dnop_v1.csv"


def main(argv: List[str]) -> None:
    # get IDs (all individual CSVs assumed to be sorted by ID_COL)
    combined: pd.DataFrame = pd.read_csv(get_csv_filepath(0)).drop(PATHOLOGIES[0], axis=1)

    singles: List[pd.DataFrame] = [
        pd.read_csv(get_csv_filepath(pi)).drop(ID_COL, axis=1)
        for pi in range(len(PATHOLOGIES))
    ]
    
    for pi in range(len(PATHOLOGIES)):
        combined = combined.join(singles[pi][PATHOLOGIES[pi]])

    combined.to_csv(COMBINED_CSV_PATH, index=False)


if __name__ == "__main__":
    main(sys.argv)
