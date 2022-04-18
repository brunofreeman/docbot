import sys
import os
from typing import List, Tuple
import numpy as np
import cv2
from progress_bar import print_progress_bar, print_progress_bar_complete


COLOR_MODE: int = cv2.IMREAD_GRAYSCALE
RESOLUTION: Tuple[int, int] = (256, 256)
INTERPOLATION: int = cv2.INTER_CUBIC

DST_DIR: str = f"{'greyscale' if COLOR_MODE == cv2.IMREAD_GRAYSCALE else 'color'}{RESOLUTION[0]}x{RESOLUTION[1]}"

TARGET_DIRS: Tuple[str] = ("train", "test", "solution")
IMG_SRC_PATH: str = "/groups/CS156b/data"
IMG_DST_PATH: str = f"/groups/CS156b/2022/team_dirs/docbot/{DST_DIR}"

N_IMAGES: int = 223413


def path_push(path: str, dir: str) -> str:
    return f"{path}/{dir}"


def path_pop(path: str) -> str:
    return path[:path.rindex("/")]


def try_mkdir_dst(src_path: str) -> None:
    os.makedirs(src_path.replace(IMG_SRC_PATH, IMG_DST_PATH), exist_ok=True)


def main(argv: List[str]) -> None:
    i: int = 0

    for split_dir in TARGET_DIRS:
        path: str = f"{IMG_SRC_PATH}/{split_dir}"

        for pid_dir in os.listdir(path):
            path = path_push(path, pid_dir)

            for study_dir in os.listdir(path):
                path = path_push(path, study_dir)
                try_mkdir_dst(path)

                for img_name in os.listdir(path):
                    print_progress_bar(i, N_IMAGES)
                    i += 1

                    path = path_push(path, img_name)

                    img: np.ndarray = cv2.imread(path, COLOR_MODE)
                    img = cv2.resize(img, dsize=RESOLUTION, interpolation=INTERPOLATION)
                    cv2.imwrite(path.replace(IMG_SRC_PATH, IMG_DST_PATH), img)
                    
                    path = path_pop(path)
                path = path_pop(path)
            path = path_pop(path)

    print_progress_bar_complete()


if __name__ == "__main__":
    main(sys.argv)
