import pandas as pd
import os

data_path = "data"
# make a list of all images
img_list = os.listdir(data_path)
# add placeholder labels for demo purposes
label_list = [0]*len(img_list)

patches = pd.DataFrame({"imgs":img_list, "labels":label_list})

csv_path = os.path.join(data_path, "patches.csv")
patches.to_csv(csv_path, index=False)
