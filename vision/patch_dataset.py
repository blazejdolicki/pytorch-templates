
from PIL import Image
from torch.utils.data import Dataset
import os

class PatchDataset(Dataset):
    def __init__(self, data_dir):
        filelist_path = os.path.join(data_dir, "patches.csv")
        # TODO init dataset

    def __get_item__(self, idx)
    img = Image.open(self.imgs[idx])

    def __len__(self):
        return len(self.imgs)
