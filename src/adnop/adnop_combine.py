import sys
from typing import List
import pandas as pd
from adnop_dataset import PATHOLOGIES
from adnop_predict import get_csv_filepath, CSV_DIR, ID_COL


def usage(argv: List[str]) -> None:
    print(f"usage: {argv[0]} -{{e, s}} <snapshot dir>", flush=True)
    sys.exit(1)


def main(argv: List[str]) -> None:
    if len(argv) != 3:
        usage(argv)

    set: str = argv[1][1]
    snapshot: str = argv[2]

    combined_csv_path: str = f"{CSV_DIR}/adnop_{set}_{snapshot}.csv"

    # get IDs (all individual CSVs assumed to be sorted by ID_COL)
    combined: pd.DataFrame = pd.read_csv(get_csv_filepath(0, set, snapshot)).drop(PATHOLOGIES[0], axis=1)

    singles: List[pd.DataFrame] = [
        pd.read_csv(get_csv_filepath(pi, set, snapshot)).drop(ID_COL, axis=1)
        for pi in range(len(PATHOLOGIES))
    ]
    
    for pi in range(len(PATHOLOGIES)):
        combined = combined.join(singles[pi][PATHOLOGIES[pi]])

    combined.to_csv(combined_csv_path, index=False)


if __name__ == "__main__":
    main(sys.argv)
