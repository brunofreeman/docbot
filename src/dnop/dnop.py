import sys
import os
import re
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from dnop_dataset import DNOPDataset, PATHOLOGIES, ViewType


BATCH_SIZE: int = 16
N_EPOCHS: int = 32

D_LOSS_THRESHOLD: float = 0.0001
N_CONSEC_DL_VIOLATIONS: int = 3

IN_DIM: Tuple[int, int, int] = (3, 256, 256)
OUT_DIM: int = 3

SAVE_DIR: str = "./out/dnop_v2"
SAVE_PATTERN: str = r"dnop_(frontal|lateral)_p[0-9]{2}_e[0-9]{3}.pt"


def get_save_filename(view_type: ViewType, pi: int, ei: int) -> str:
    return f"dnop_{view_type}_p{pi:02d}_e{ei:03d}.pt"


def get_save_filepath(view_type: ViewType, pi: int, ei: int) -> str:
    return f"{SAVE_DIR}/{get_save_filename(view_type, pi, ei)}"


def is_save_filename(filename: str) -> bool:
    return re.fullmatch(SAVE_PATTERN, filename)


def extract_params(save_name: str) -> Tuple[ViewType, int, int]:
    vt = ViewType.FRONTAL if str(ViewType.FRONTAL) in save_name else ViewType.LATERAL
    pi = int(save_name[-10:-8])
    epoch = int(save_name[-6:-3])
    return vt, pi, epoch


def usage(argv: List[str]) -> None:
    print(f"usage: {argv[0]} {{f, l}} <i \u2208 [0, 13]>", flush=True)
    sys.exit(1)


def get_model(device: torch.device) -> nn.Sequential:
    model = torchvision.models.densenet121(pretrained=True) 
    model.classifier = nn.Sequential(
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, OUT_DIM)
        # cross-entropy loss will apply Softmax for us
    )
    return model.to(device)


def parse_args(argv: List[str]) -> Tuple[ViewType, int]:
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
    
    return view_type, pi


def most_recent_save(view_type: ViewType, pi: int) -> Optional[str]:
    ei: int = 0
    filename = None
    for pt_file in os.listdir(SAVE_DIR):
        if is_save_filename(pt_file):
            v, p, e = extract_params(pt_file)
            if v == view_type and p == pi and ei < e:
                ei = e
                filename = pt_file
    return filename


def main(argv: List[str]) -> None:
    view_type, pi = parse_args(argv)

    header: str = f"DNOP v2 Model for Pathology {pi:02d} ({PATHOLOGIES[pi]}) -- {str(view_type).capitalize()} View"
    print(f"{header}\n{'=' * len(header)}\n", flush=True)

    fname: Optional[str] = most_recent_save(view_type, pi)
    epochs_already = 0 if fname is None else extract_params(fname)[2]
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_loader = DataLoader(
        DNOPDataset(view_type, pi, device), batch_size=BATCH_SIZE, shuffle=True
    )

    model = get_model(device)

    if epochs_already > 0:
        load_path: str = get_save_filepath(view_type, pi, epochs_already)
        model.load_state_dict(torch.load(load_path, map_location=device))
        print(f"Loaded {load_path}", flush=True)
    
    print()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    best_loss: Optional[float] = None
    ncv: int = 0

    if epochs_already >= N_EPOCHS:
        print(f"all {N_EPOCHS} training epochs already complete, exiting...", flush=True)

    for ei in range(epochs_already, N_EPOCHS):
        title: str = f"Epoch {(ei + 1):03d}/{(N_EPOCHS):03d}:"
        print(f"{title}\n{'-' * len(title)}", flush=True)

        training_loss: float = 0.0

        model.train()
        for _, (images, labels) in enumerate(data_loader):
            # erase acculmulated gradients
            optimizer.zero_grad()

            # forward pass
            output = model(images)

            # copute loss
            loss = criterion(output, labels)

            # backward pass
            loss.backward()

            # update weights
            optimizer.step()

            # track training loss
            training_loss += loss.item()

        # update loss and training accuracy
        training_loss /= len(data_loader)
        
        # save model
        save_path = get_save_filepath(view_type, pi, ei + 1)
        torch.save(model.state_dict(), save_path)

        # remove save from previous epoch to save space
        if ei > 0:
            old_save_path: str = get_save_filepath(view_type, pi, ei)
            if os.path.exists(old_save_path):
                os.remove(old_save_path)

        print(f"loss: {training_loss:0.8f}", flush=True)
        print(f"model saved to {save_path}", flush=True)
        print()

        if best_loss is not None and best_loss - training_loss < D_LOSS_THRESHOLD:
            ncv += 1
            if ncv == N_CONSEC_DL_VIOLATIONS:
                print("terminating early due to loss stagnation...", flush=True)
                break
        else:
            best_loss = training_loss
            ncv = 0


if __name__ == "__main__":
    main(sys.argv)
