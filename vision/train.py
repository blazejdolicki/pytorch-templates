from logging import root
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import random
import numpy as np

from net import Net
from patch_dataset import PatchDataset


# arguments
batch_size = 16
data_dir = "C:/Users/BlazejDolicki/Documents/Personal/Git/pytorch-templates/vision/data"
num_classes = 10
learning_rate = 0.001
momentum = 0.9
num_epochs = 2
checkpoint_dir = "checkpoints"
seed = 7

os.makedirs(checkpoint_dir, exist_ok=True)

# fix seed
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

fix_seed(seed)


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5,),
                                                     (0.5, 0.5, 0.5))])


train_dataset = PatchDataset(data_dir=data_dir, split="train", transform=transform)
val_dataset = PatchDataset(data_dir=data_dir, split="valid", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = Net(num_classes=num_classes).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

criterion = torch.nn.CrossEntropyLoss()

for epoch in tqdm(range(num_epochs)):
    model.train()

    running_loss = 0.0
    num_correct = 0.0

    for inputs, labels in tqdm(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # reset gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(inputs)

        # calculate loss
        loss = criterion(outputs, labels)

        # backward pass computes gradients
        loss.backward()

        # optimizer updates the parameters
        optimizer.step()

        # update running loss
        actual_batch_size = inputs.shape[0]
        running_loss += loss.item() * actual_batch_size

        # update running accuracy
        preds = outputs.argmax(dim=1)
        num_correct += (preds == labels).sum().item()

    # evaluate on validation set after every epoch
    model.eval()

    val_running_loss = 0.0
    val_num_correct = 0.0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)  

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # update validation running loss
            actual_batch_size = inputs.shape[0]
            val_running_loss += loss.item() * actual_batch_size

            # update validation accuracy
            preds = outputs.argmax(dim=1)
            val_num_correct += (preds == labels).sum().item()


    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = 100 * num_correct / len(train_dataset)

    val_epoch_loss = val_running_loss / len(val_dataset)
    val_epoch_acc = 100 * val_num_correct / len(val_dataset)

    print(f"Epoch: {epoch}")
    print(f"train loss: {epoch_loss}, train accuracy: {epoch_acc}")
    print(f"val loss: {val_epoch_loss}, val accuracy {val_epoch_acc}")

    # save model
    model_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch}.pt")
    torch.save(model.state_dict(), model_path)







