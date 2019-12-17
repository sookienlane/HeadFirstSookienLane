import torch
import os
from glob import glob
from PIL import Image
import numpy as np

class ImageTrainDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform):
        file_path = glob(os.path.join(data_path, "*/images/*"))
        label_raw = [x.split("/")[-1].split("_")[0] for x in file_path]
        self.file_path = file_path
        self.transforms = transform
        self.label2index_dict = {v: k for k, v in enumerate(np.unique(label_raw))}
        self.index2label_dict = {k: v for k, v in enumerate(np.unique(label_raw))}
        self.labels = [self.label2index_dict.get(x) for x in label_raw]

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, index):
        image = Image.open(self.file_path[index]).convert('RGB')
        image = self.transforms(image)
        label = self.labels[index]
        return {"feature": image, "label": label}

class ImageValDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, transform,label2index_dict):
        file_path = glob(os.path.join(data_path, "images/*"))
        file2label_dict = {x.split("\t")[0]:x.split("\t")[1] for x in open(os.path.join(data_path, "val_annotations.txt"))}
        label_raw = [file2label_dict.get(x.split("/")[-1])for x in file_path]
        self.file_path = file_path
        self.transforms = transform
        self.labels = [label2index_dict.get(x) for x in label_raw]

    def __len__(self):
        return len(self.file_path)

    def __getitem__(self, index):
        image = Image.open(self.file_path[index]).convert('RGB')
        image = self.transforms(image)
        label = self.labels[index]
        return {"feature": image, "label": label}