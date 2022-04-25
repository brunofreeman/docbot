import sys
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from dataset_simple import CheXpertTrainingDataset

BATCH_SIZE: int = 32
N_EPOCHS: int = 128

IN_DIM: Tuple[int, int, int] = (1, 256, 256)
OUT_DIM: int = 14

def main(argv: List[str]) -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    data_loader = DataLoader(
        CheXpertTrainingDataset(device), batch_size=BATCH_SIZE, shuffle=True
    )

    model = torchvision.models.densenet121(pretrained=True)
    first_conv_layer = [nn.Conv2d(1, 3, kernel_size=(3,3))]
    first_conv_layer.extend(list(model.features))  
    model.features= nn.Sequential(*first_conv_layer ) 
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, OUT_DIM)
    ) 
    model = model.to(device)
    print(model)

    # BCEWithLogitsLoss for a multi-class classification
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.RMSprop(model.parameters())

    # store metrics
    # training_accuracy_history = np.zeros([N_EPOCHS, 1])
    # training_loss_history = np.zeros([N_EPOCHS, 1])
    # validation_accuracy_history = np.zeros([N_EPOCHS, 1])
    # validation_loss_history = np.zeros([N_EPOCHS, 1])

    for epoch_idx in range(N_EPOCHS):
        title: str = f"Epoch {epoch_idx+1:03d}/{N_EPOCHS}:"
        print(f"{title}\n{'-' * len(title)}")

        model.train()
        for batch_idx, (images, labels) in enumerate(data_loader):
            print(type(images))
            print(images)
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
        
        save_path: str = f"./out/chexnet_v1_{(epoch_idx + 1):03d}.pt"
        print(f"Saving model to {save_path}")
        torch.save(model.state_dict(), save_path)
        
        # TODO @mbatchev
        """   
            # track training accuracy
            _, predicted = torch.max(output.data, 1)
            train_total += labels.size(0)
            print(predicted)
            train_correct += (predicted == labels).sum().item()
            # track training loss
            training_loss_history[epoch_idx] += loss.item()
            # progress update after 180 batches (~1/10 epoch for batch size 32)
            if i % 180 == 0: print('.',end='')
       
        train_total = 0
        train_correct = 0

        training_loss_history[epoch_idx] /= len(data_loader)
        training_accuracy_history[epoch_idx] = train_correct / train_total
        print(f'\n\tloss: {training_loss_history[epoch,0]:0.4f}, acc: {training_accuracy_history[epoch,0]:0.4f}',end='')
            
        # validate
        test_total = 0
        test_correct = 0
        with torch.no_grad():
            model.eval()
            for i, data in enumerate(data_loader):
                images, labels = data
                # forward pass
                output = model(images)
                # find accuracy
                _, predicted = torch.max(output.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                # find loss
                loss = criterion(output, labels)
                validation_loss_history[epoch_idx] += loss.item()
            validation_loss_history[epoch_idx] /= len(data_loader)
            validation_accuracy_history[epoch_idx] = test_correct / test_total
        print(f', val loss: {validation_loss_history[epoch_idx, 0]:0.4f}, val acc: {validation_accuracy_history[epoch_idx, 0]:0.4f}')
        """


if __name__ == "__main__":
    main(sys.argv)