import pandas as pd
import os
import numpy as np

# Create dummy csv files for training and validation set containing images and corresponding labels

data_path = "data"

if "train_patches.csv" not in os.listdir("data") and "valid_patches.csv" not in os.listdir("data"):
    # make a list of all images
    img_list = os.listdir(data_path)
    train_img_list = img_list[:8]
    valid_img_list = img_list[8:]
    for img_list, split in zip([train_img_list, valid_img_list], ["train", "valid"]):
        # add placeholder labels for demo purposes
        label_list = [0]*len(img_list)

        patches = pd.DataFrame({"imgs":img_list, "labels":label_list})
        csv_path = os.path.join(data_path, f"{split}_patches.csv")
        patches.to_csv(csv_path, index=False)
else:
    print("Training or validation csv files already created.")