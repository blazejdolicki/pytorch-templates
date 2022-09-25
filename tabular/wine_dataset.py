
import pandas as pd
from torch.utils.data import Dataset
import torch


class WineDataset(Dataset):
    def __init__(self, data_path):

        data = pd.read_csv(data_path, sep=";")

        self.inputs = torch.tensor(data.iloc[:, :-1].values)
        self.labels = torch.tensor(data.iloc[:,-1].values)

        del data

    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


if __name__ == "__main__":
    from wine_dataset import WineDataset

    data_path = "D:/Blazej/Dokumenty/data/train_wine.csv"
    dataset = WineDataset(data_path)

    print("Dataset size", len(dataset))
    print("First element")
    print(dataset[0])
