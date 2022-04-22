import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataset_simple import CheXpertTrainingDataset

# transforms.ToTensor() converts batch of images to 4-D tensor and normalizes 0-255 to 0-1.0

BATCH_SIZE = 32
dataset = CheXpertTrainingDataset()

training_data_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True)


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
    nn.Linear(2592, 64), #find the number this should be
    nn.ReLU(),
    nn.Linear(64, 15)
    # PyTorch implementation of cross-entropy loss includes softmax layer
)


# For a multi-class classification problem
# TODO: This wont work when multiple classes can be ture
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCEWithLogitsLoss()

optimizer = optim.RMSprop(model.parameters())

n_epochs = 10

# store metrics
training_accuracy_history = np.zeros([n_epochs, 1])
training_loss_history = np.zeros([n_epochs, 1])
validation_accuracy_history = np.zeros([n_epochs, 1])
validation_loss_history = np.zeros([n_epochs, 1])

for epoch in range(n_epochs):
    print(f'Epoch {epoch+1}/10:', end='')
    train_total = 0
    train_correct = 0
    # train
    model.train()
    for i, data in enumerate(training_data_loader):
        images, labels = data
        optimizer.zero_grad()
        # forward pass
        output = model(images)
        # calculate categorical cross entropy loss
        loss = criterion(output, labels)
        # backward pass
        loss.backward()
        optimizer.step()
        
        # track training accuracy
        _, predicted = torch.max(output.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        # track training loss
        training_loss_history[epoch] += loss.item()
        # progress update after 180 batches (~1/10 epoch for batch size 32)
        if i % 180 == 0: print('.',end='')
    training_loss_history[epoch] /= len(training_data_loader)
    training_accuracy_history[epoch] = train_correct / train_total
    print(f'\n\tloss: {training_loss_history[epoch,0]:0.4f}, acc: {training_accuracy_history[epoch,0]:0.4f}',end='')
        
    # validate
    test_total = 0
    test_correct = 0
    # with torch.no_grad():
    #     model.eval()
    #     for i, data in enumerate(test_data_loader):
    #         images, labels = data
    #         # forward pass
    #         output = model(images)
    #         # find accuracy
    #         _, predicted = torch.max(output.data, 1)
    #         test_total += labels.size(0)
    #         test_correct += (predicted == labels).sum().item()
    #         # find loss
    #         loss = criterion(output, labels)
    #         validation_loss_history[epoch] += loss.item()
    #     validation_loss_history[epoch] /= len(test_data_loader)
    #     validation_accuracy_history[epoch] = test_correct / test_total
    # print(f', val loss: {validation_loss_history[epoch,0]:0.4f}, val acc: {validation_accuracy_history[epoch,0]:0.4f}')
