# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 17:59:23 2020

@author: Steff
"""

import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch

class Dataset:

    ##########################################################################
    def __init__(
            self,
            folder           : str,
            batch_size       : int,
            worker           : int = 2,
            shuffle          : bool = True
        ):
        
        preprocess = [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
        
        dataset = dset.ImageFolder(
            root = folder,
            transform = transforms.Compose(preprocess)
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle = shuffle,
            pin_memory = True,
            num_workers = worker
        )
        
        self.dataset = dataset
        self.length = len(self.dataset)
        self.dataloader = dataloader