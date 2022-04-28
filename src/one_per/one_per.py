import sys
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary
from one_per_dataset import CheXpertOnePerDataset, PATHOLOGIES

BATCH_SIZE: int = 32
N_EPOCHS: int = 16

IN_DIM: Tuple[int, int, int] = (1, 256, 256)
OUT_DIM: int = 1

def usage(argv: List[str]) -> None:
    print(f"usage: {argv[0]} <i \u2208 [0, 13]>")
    sys.exit(1)

def main(argv: List[str]) -> None:
    try:
        pi: int = int(argv[1])
    except (IndexError, ValueError):
        usage(argv)
    
    if not (0 <= pi < len(PATHOLOGIES)):
        usage(argv)

    header: str = f"One-Per Model for Pathology {pi} ({PATHOLOGIES[pi]})"
    print(f"{header}\n{'=' * len(header)}\n")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    data_loader = DataLoader(
        CheXpertOnePerDataset(pi, device), batch_size=BATCH_SIZE, shuffle=True
    )

    model = nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=(3,3)),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(p=0.1),
        
        nn.Conv2d(8, 16, kernel_size=(3,3)),
        nn.BatchNorm2d(16),
        nn.ReLU(),

        nn.Conv2d(16, 32, kernel_size=(3,3)),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Dropout(p=0.1),
        
        nn.Flatten(),
        nn.Linear(484128, 64),
        nn.ReLU(),
        nn.Linear(64, OUT_DIM),
        nn.Tanh()
    ).to(device)

    print("Model:")
    print(model)
    summary(model, IN_DIM)
    print()

    criterion = nn.MSLELoss()

    optimizer = optim.RMSprop(model.parameters())

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
        save_path: str = f"./out/one_per_p{pi:02d}_e{(epoch_idx + 1):03d}.pt"
        torch.save(model.state_dict(), save_path)

        print(f"loss: {training_loss:0.4f}")
        print(f"model saved to {save_path}")
        print()


if __name__ == "__main__":
    main(sys.argv)
