import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

class TorchTabularTextDataset(TorchDataset):
    def __init__(self,
                 encodings,
                 categorical_feats,
                 numerical_feats,
                 labels=None,
                 df=None,
                 label_list=None,
                 class_weights=None
                 ):
        self.df = df
        self.encodings = encodings
        self.cat_feats = categorical_feats
        self.numerical_feats = numerical_feats
        self.labels = labels
        self.class_weights = class_weights
        self.label_list = label_list if label_list is not None else [i for i in range(len(np.unique(labels)))]

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]) if self.labels is not None  else None
        item['cat_feats'] = torch.tensor(self.cat_feats[idx]).float() \
            if self.cat_feats is not None else torch.zeros(0)
        item['numerical_feats'] = torch.tensor(self.numerical_feats[idx]).float()\
            if self.numerical_feats is not None else torch.zeros(0)
        return item

    def __len__(self):
        return len(self.labels)

    def get_labels(self):
        return self.label_list
