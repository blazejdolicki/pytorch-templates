import pandas as pd
import numpy as np
import random

data_path = "D:\Blazej\Dokumenty\data\winequality-white.csv"
train_path = "D:/Blazej/Dokumenty/data/train_wine.csv"
val_path = "D:/Blazej/Dokumenty/data/val_wine.csv"

data = pd.read_csv(data_path, sep=";")
num_examples = data.shape[0]
train_fraction = 0.8
train_idxs =  random.sample(range(num_examples), int(num_examples*train_fraction))

all_idxs = np.array(range(num_examples))
val_idxs = np.setdiff1d(all_idxs, train_idxs)


train = data.iloc[train_idxs]
train.to_csv(train_path, index=False, sep=";")

val = data.iloc[val_idxs]
val.to_csv(val_path, index=False, sep=";")
