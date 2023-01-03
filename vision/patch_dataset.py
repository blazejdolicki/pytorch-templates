
from PIL import Image
from torch.utils.data import Dataset
import os
import pandas as pd

class PatchDataset(Dataset):
    def __init__(self, data_dir, split, transform=None):
        csv_path = os.path.join(data_dir, f"{split}_patches.csv")
        data = pd.read_csv(csv_path)
        # get list of image paths
        self.imgs = [os.path.join(data_dir, img) for img in data["imgs"].values]
        # get list of labels corresponding to images
        self.labels = data["labels"].values

        self.transform = transform

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx])
        label = self.labels[idx]
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)
