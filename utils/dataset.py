import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from utils.transforms import inference_transform

class FERDataset(Dataset):

    def __init__(self, csv_file, transform=None):

        self.data = pd.read_csv(csv_file)
        self.transform = transform if transform is not None else inference_transform

        # 只保留四类
        self.keep_labels = [0, 3, 4, 6]

        self.data = self.data[self.data['emotion'].isin(self.keep_labels)]

        self.label_map = {
            0:0,  # angry
            3:1,  # happy
            4:2,  # sad
            6:3   # neutral
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        pixels = np.fromstring(row['pixels'], sep=' ')
        image = pixels.reshape(48, 48).astype(np.uint8)

        # 转为 PIL Image 以便使用 transforms 管线
        image = Image.fromarray(image, mode='L')

        image = self.transform(image)

        label = self.label_map[row['emotion']]

        label = torch.tensor(label)

        return image, label