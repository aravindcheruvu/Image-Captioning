import json

import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from torch.utils.data import random_split

from FlickrJSON import FlickrJSON




class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""

    def __init__(self, root, json_data, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        self.pythonObj = json.loads(json_data)
        self.ids = []
        for i in range(len(self.pythonObj['annotations'])):
            self.ids.append(i)
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        vocab = self.vocab
        ann_id = self.ids[index]
        caption = str(self.pythonObj['annotations'][index]['caption'])
        img_id = self.pythonObj['annotations'][index]['image_id']
        path = self.pythonObj['annotations'][index]['file_name']
        # print(path)
        # print(os.path.join(self.root, path))
        file_name = path
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target,file_name

    def __len__(self):
        return len(self.ids)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions,file_name  = zip(*data)
    # print(captions)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths,file_name


def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    flickj = FlickrJSON()
    JSON_data = flickj.BuildJson(json)
    coco = CocoDataset(root=root,
                       json_data = JSON_data,
                       vocab=vocab,
                       transform=transform)

    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    # trainset_2 = torch.utils.data.Subset(coco,[0,1,2,3,4,5,6,7,8,9])
    # data_loader = torch.utils.data.DataLoader(dataset=trainset_2,
    #                                           batch_size=batch_size,
    #                                           shuffle=shuffle,
    #                                           num_workers=num_workers,
    #                                           collate_fn=collate_fn)


    trainval_size = int(0.9 * len(coco))
    test_size = len(coco) - trainval_size

    train_size = int(0.888*trainval_size)
    val_size = trainval_size - train_size

    trainval_dataset, TestDataset = random_split(coco,[trainval_size,test_size] )
    TrainDataset, ValidationDataset = random_split(trainval_dataset,[train_size,val_size] )

    Traindata_loader = torch.utils.data.DataLoader(dataset=TrainDataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    Validationdata_loader = torch.utils.data.DataLoader(dataset=ValidationDataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    Testdata_loader = torch.utils.data.DataLoader(dataset=TestDataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return Traindata_loader,Validationdata_loader,Testdata_loader