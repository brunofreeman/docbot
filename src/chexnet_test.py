import sys
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from dataset_chexnet import CheXpertTrainingDataset

BATCH_SIZE: int = 32
N_EPOCHS: int = 128

IN_DIM: Tuple[int, int, int] = (3, 224, 224)
OUT_DIM: int = 14

def main(argv: List[str]) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    data_loader = DataLoader(
        CheXpertTrainingDataset(device), batch_size=BATCH_SIZE, shuffle=True
    )

    model = torchvision.models.densenet121(pretrained=True) 
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, OUT_DIM),
            nn.Tanh()
    ) 
    model = model.to(device)
    print(model)

    # BCEWithLogitsLoss for a multi-class classification
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.RMSprop(model.parameters())


    for epoch_idx in range(N_EPOCHS):
        title: str = f"Epoch {epoch_idx+1:03d}/{N_EPOCHS}:"
        print(f"{title}\n{'-' * len(title)}")
        training_loss: float = 0.0
        model.train()

        for batch_idx, (images, labels) in enumerate(data_loader):
            #print(images)
            # erase acculmulated gradients
            optimizer.zero_grad()

            # forward pass
            output = model(images)

            # compute loss
            loss = criterion(output, labels)

            # backward pass
            loss.backward()

            # update weights
            optimizer.step()

            training_loss += loss.item()
            print(loss)

        # update loss and training accuracy
        training_loss /= len(data_loader)

        save_path: str = f"./out/chexnet_v1_{(epoch_idx + 1):03d}.pt"
        f = open("./out/chexnet_test_live.txt", "w")
        f.write(f"loss: {training_loss:0.4f}")
        f.close()
        print(f"loss: {training_loss:0.4f}")
        print(f"Saving model to {save_path}")
        torch.save(model.state_dict(), save_path)
     

if __name__ == "__main__":
    main(sys.argv)
