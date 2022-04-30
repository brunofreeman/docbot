import sys
import os
from typing import List, Tuple
import numpy as np
import cv2

# options: cv2.IMREAD_{COLOR, GRAYSCALE}
COLOR_MODE: int = cv2.IMREAD_COLOR
RESOLUTION: Tuple[int, int] = (512, 512)
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
    if os.path.exists(IMG_DST_PATH):
        print(f"{DST_DIR} has already been generated!")
        print(f"If you are regenerating, delete the directory first with 'rm -rf {DST_DIR}' while in {IMG_DST_PATH[:IMG_DST_PATH.rindex('/')]}.")
        print(f"Be warned that regenerating requires many hours and a Slurm job, so don't do so unless necessary.")
        sys.exit(1)

    for split_dir in TARGET_DIRS:
        path: str = f"{IMG_SRC_PATH}/{split_dir}"

        for pid_dir in os.listdir(path):
            path = path_push(path, pid_dir)

            for study_dir in os.listdir(path):
                path = path_push(path, study_dir)
                try_mkdir_dst(path)

                for img_name in os.listdir(path):
                    path = path_push(path, img_name)

                    img: np.ndarray = cv2.imread(path, COLOR_MODE)
                    img = cv2.resize(img, dsize=RESOLUTION, interpolation=INTERPOLATION)
                    cv2.imwrite(path.replace(IMG_SRC_PATH, IMG_DST_PATH), img)
                    
                    path = path_pop(path)
                path = path_pop(path)
            path = path_pop(path)


if __name__ == "__main__":
    main(sys.argv)
