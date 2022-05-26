import timm
import sys
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from xception_dataset import CheXpertTrainingDataset

BATCH_SIZE: int = 16
N_EPOCHS: int = 128

IN_DIM: Tuple[int, int, int] = (3, 512, 512)
OUT_DIM: int = 14


def main(argv: List[str]) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    data_loader = DataLoader(
        CheXpertTrainingDataset(device), batch_size=BATCH_SIZE, shuffle=True
    )


    # model comes from timm; it has a nice feature where we can use num_classes to
    # adjust model.fc on our own
    model = timm.create_model('xception', pretrained=False, num_classes = OUT_DIM)

    model = model.to(device)
    print(model)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch_idx in range(N_EPOCHS):

        f = open("./out/xception_test_live_smaller.txt", "a")
        title: str = f"Epoch {epoch_idx+1:03d}/{N_EPOCHS}:"
        print(f"{title}\n{'-' * len(title)}")
        f.write(f"{title}\n{'-' * len(title)}\n")
        f.close()
        training_loss: float = 0.0
        model.train()

        for batch_idx, (images, labels) in enumerate(data_loader):
            #print(images)
            # erase acculmulated gradients
            optimizer.zero_grad()

            # forward pass

            output = model(images)
            # print("images shape", images.shape)
            # print("output shape", output.shape)
            # print("labels shape", labels.shape)
            # compute loss
            loss = criterion(output, labels)

            # backward pass
            loss.backward()

            # update weights
            optimizer.step()

            training_loss += loss.item()

        # update loss and training accuracy
        training_loss /= len(data_loader)

        save_path: str = f"./out/xception_v1_smaller_{(epoch_idx + 1):03d}.pt"
        f = open("./out/xception_test_live_smaller.txt", "a")
        f.write(f"loss: {training_loss:0.4f}\n")
        f.close()
        print(f"loss: {training_loss:0.4f}")
        print(f"Saving model to {save_path}")
        torch.save(model.state_dict(), save_path)   
    f.close()    
     

if __name__ == "__main__":
    main(sys.argv)
