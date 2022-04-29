import sys
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary
from one_per_dataset import CheXpertOnePerDataset, PATHOLOGIES, ViewType


BATCH_SIZE: int = 32
N_EPOCHS: int = 16

IN_DIM: Tuple[int, int, int] = (1, 256, 256)
OUT_DIM: int = 1


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

    header: str = f"One-Per Model for Pathology {pi:02d} ({PATHOLOGIES[pi]})"
    print(f"{header}\n{'=' * len(header)}\n")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_loader = DataLoader(
        CheXpertOnePerDataset(view_type, pi, device), batch_size=BATCH_SIZE, shuffle=True
    )

    model = nn.Sequential(
        # in: 1 256x256 channel
        nn.Conv2d(1, 128, kernel_size=32+1, padding="same"),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(p=0.1),
        
        # in: 128 128x128 channels
        nn.Conv2d(128, 128, kernel_size=32+1, padding="same"),
        nn.BatchNorm2d(128),
        nn.ReLU(),

        # in: 128 128x128 channels
        nn.Conv2d(128, 64, kernel_size=16+1, padding="same"),
        nn.BatchNorm2d(64),
        nn.ReLU(),

        # in: 64 128x128 channels
        nn.Conv2d(64, 64, kernel_size=16),
        nn.BatchNorm2d(64),
        nn.ReLU(),

        # in: 64 113x113 channels (113 = 128 - 16 + 1)
        nn.Conv2d(64, 64, kernel_size=8),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Dropout(p=0.1),
        
        # in: 64 106x106 channels (106 = 113 - 8 + 1)
        nn.Conv2d(64, 32, kernel_size=8),
        nn.BatchNorm2d(32),
        nn.ReLU(),

        # in: 32 99x99 channels (99 = 106 - 8 + 1)
        nn.Flatten(),
        nn.Linear(32 * 99 * 99, 256),
        nn.ReLU(),
        nn.Linear(256, OUT_DIM),
        nn.Tanh()
    ).to(device)

    print("Model:")
    print(model)
    summary(model, IN_DIM)
    print()

    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters())

    for epoch_idx in range(N_EPOCHS):
        title: str = f"Epoch {epoch_idx+1:03d}/{N_EPOCHS:03d}:"
        print(f"{title}\n{'-' * len(title)}")

        training_loss: float = 0.0

        model.train()
        for bi, (images, labels) in enumerate(data_loader):
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
        save_path: str = f"./out/one_per_v2/one_per_p{pi:02d}_e{(epoch_idx + 1):03d}.pt"
        torch.save(model.state_dict(), save_path)

        print(f"loss: {training_loss:0.4f}")
        print(f"model saved to {save_path}")
        print()


if __name__ == "__main__":
    main(sys.argv)
