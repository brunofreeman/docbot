import sys
import os
import re
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from dnop_dataset import DNOPDataset, PATHOLOGIES, ViewType


BATCH_SIZE: int = 32
N_EPOCHS: int = 3

IN_DIM: Tuple[int, int, int] = (3, 256, 256)
OUT_DIM: int = 3

OUT_DIR: str = "./out/dnop"
SAVE_PATTERN: str = r"dnop_(frontal|lateral)_p[0-9]{2}_e[0-9]{3}.pt"


def get_save_filename(view_type: ViewType, pi: int, ei: int) -> str:
    return f"dnop_{view_type}_p{pi:02d}_e{ei:03d}.pt"


def get_save_filepath(view_type: ViewType, pi: int, ei: int) -> str:
    return f"{OUT_DIR}/{get_save_filename(view_type, pi, ei)}"


def is_save_filename(filename: str) -> bool:
    return re.fullmatch(SAVE_PATTERN, filename)


def extract_epoch(save_name: str) -> int:
    return int(save_name[-6:-3])


def usage(argv: List[str]) -> None:
    print(f"usage: {argv[0]} {{f, l}} <i \u2208 [0, 13]>")
    sys.exit(1)


def get_model(device: torch.device) -> nn.Sequential:
    model = torchvision.models.densenet121(pretrained=True) 
    model.classifier = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, OUT_DIM)
        # cross-entropy loss will apply Softmax for us
    )
    return model.to(device)


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

    header: str = f"DNOP Model for Pathology {pi:02d} ({PATHOLOGIES[pi]})"
    print(f"{header}\n{'=' * len(header)}\n")

    epoch_idx: int = 0

    for pt_file in os.listdir(OUT_DIR):
        if is_save_filename(pt_file):
            epoch_idx = max(epoch_idx, extract_epoch(pt_file))
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_loader = DataLoader(
        DNOPDataset(view_type, pi, device), batch_size=BATCH_SIZE, shuffle=True
    )

    model = get_model(device)

    print("Model:")
    print(model)

    if epoch_idx > 0:
        load_path: str = get_save_filepath(view_type, pi, epoch_idx)
        model.load_state_dict(torch.load(load_path, map_location=device))
        print(f"Loaded {load_path}")
    
    print()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    init_epochs: int = epoch_idx

    for _ in range(N_EPOCHS):
        title: str = f"Epoch {(epoch_idx + 1):03d}/{(init_epochs + N_EPOCHS):03d}:"
        print(f"{title}\n{'-' * len(title)}")

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
        save_path = get_save_filepath(view_type, pi, epoch_idx + 1)
        torch.save(model.state_dict(), save_path)

        # remove save from previous epoch to save space
        if epoch_idx > 0:
            old_save_path: str = get_save_filepath(view_type, pi, epoch_idx)
            if os.path.exists(old_save_path):
                os.remove(old_save_path)

        print(f"loss: {training_loss:0.8f}")
        print(f"model saved to {save_path}")
        print()

        epoch_idx += 1


if __name__ == "__main__":
    main(sys.argv)
