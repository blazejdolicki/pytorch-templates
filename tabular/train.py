import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from net import Net
from wine_dataset import WineDataset

# arguments
train_path = "D:/Blazej/Dokumenty/data/train_wine.csv"
val_path = "D:/Blazej/Dokumenty/data/val_wine.csv"
batch_size = 64
num_inputs = 11
num_hidden = 40
num_epochs = 3
learning_rate = 0.0001
momentum = 0.9
# because we're doing regression
num_outputs = 1


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# initialize data loaders
train_dataset = WineDataset(train_path)
val_dataset = WineDataset(val_path)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# initialize the network
model = Net(num_inputs=num_inputs, num_hidden=num_hidden, num_outputs=num_outputs).to(device)

# initialize Mean Squared Error loss because we're doing regression
criterion = torch.nn.MSELoss()

# initialize the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# training loop
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for inputs, labels in train_loader:
        # move tensors to GPU
        inputs = inputs.float().to(device)
        labels = labels.float().to(device)

        print("inputs type", inputs.dtype, inputs.shape)
        print("labels type", labels.dtype)

        # forward pass through the model
        preds = model(inputs)

        # reset the gradient before computing it
        optimizer.zero_grad()

        # compute the loss
        loss = criterion(preds, labels)

        print("Loss", loss)

        # backward pass computes the gradient
        loss.backward()

        optimizer.step()













