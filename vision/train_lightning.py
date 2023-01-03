import os
import pytorch_lightning as pl
from net import Net
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchmetrics

from patch_dataset import PatchDataset
from torch.utils.data import DataLoader

# arguments
batch_size = 16
data_dir = "C:/Users/BlazejDolicki/Documents/Personal/Git/pytorch-templates/vision/data" # TODO: Adjust this for your directory
num_classes = 10
learning_rate = 0.001
momentum = 0.9
num_epochs = 2
checkpoint_dir = "checkpoints"
seed = 7

os.makedirs(checkpoint_dir, exist_ok=True)

class LitNet(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.model = Net(num_classes=num_classes)
        self.train_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
    
    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        inputs, labels = train_batch
        outputs = self.forward(inputs)
        loss = F.cross_entropy(outputs, labels)
        self.log("train_loss", loss, on_epoch=True)
        self.train_accuracy(outputs, labels)
        self.log("train_acc", self.train_accuracy, on_epoch=True)
        return loss
    
    def validation_step(self, valid_batch, batch_idx):
        inputs, labels = valid_batch
        outputs = self.forward(inputs)
        loss = F.cross_entropy(outputs, labels)
        self.log("val_loss", loss, on_epoch=True)
        self.val_accuracy(outputs, labels)
        self.log("val_acc", self.val_accuracy, on_epoch=True)
        return loss


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5,),
                                                     (0.5, 0.5, 0.5))])


train_dataset = PatchDataset(data_dir=data_dir, split="train", transform=transform)
val_dataset = PatchDataset(data_dir=data_dir, split="valid", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = LitNet(num_classes=num_classes)
trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=num_epochs)
trainer.fit(model, train_loader, val_loader)

    