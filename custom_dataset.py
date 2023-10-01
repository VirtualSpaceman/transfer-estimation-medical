#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd

from PIL import Image

import torch
from torch.utils.data import Dataset

class CSVDataset(Dataset):
    def __init__(self, imgs_folder, labels_csv, 
                 _format = '.png', sep = ',', transforms=None):
        super(CSVDataset, self).__init__()
        self.imgs_folder = imgs_folder
        self.labels_csv = labels_csv
        self.augmentations = transforms
        self._format = _format
        
        self.csv = pd.read_csv(labels_csv, sep=sep)
        
        # Calculate class weights for WeightedRandomSampler
        self.class_counts = dict(self.csv['label'].value_counts())
        self.class_weights = {label: max(self.class_counts.values()) / count
                              for label, count in self.class_counts.items()}
        self.sampler_weights = [self.class_weights[cls]
                                for cls in self.csv['label']]
        self.class_weights_list = [self.class_weights[k]
                                   for k in sorted(self.class_weights)]

        classes = list(self.csv['label'].unique())
        classes.sort()
        self.class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.classes = classes

        print('Found {} images from {} classes.'.format(len(self.csv),
                                                        len(classes)))
        for class_name, idx in self.class_to_idx.items():
            n_images = dict(self.csv['label'].value_counts())
            print("    Class '{}' ({}): {} images.".format(
                class_name, idx, n_images[class_name]))
    
    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, idx):
        csv_line = self.csv.iloc[idx]
        img_filename = csv_line['image'] + self._format
        img_path = os.path.join(self.imgs_folder, img_filename)
        img = Image.open(img_path).convert("RGB")
        
        label = torch.tensor(int(csv_line['label']))
        
        if self.augmentations is not None:
            img = self.augmentations(img)
        
        return img, label
    
class CSVDatasetWithName(CSVDataset):
    def __getitem__(self, idx):
        name = self.csv.iloc[idx].image
        return super().__getitem__(idx), name
    
    