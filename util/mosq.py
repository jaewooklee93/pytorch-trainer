import io
import gc
import traceback
import subprocess

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fastprogress.fastprogress import master_bar, progress_bar

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
# from copy import deepcopy 

"""
Models
"""

class BaseModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        if isinstance(model, str):
            self = torch.load(model, map_location='cpu')
        else:
            self.model = model
        
    def forward(self, x):
        return self.model(x)
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def save(self, filename):
        device = self.device
        self.to('cpu')
        torch.save(self, filename)
        self.to(device)
        

class SegmentationModel(BaseModel):
    def step(self, batch, criterion):
        img, mask = batch["image"], (batch["mask"] > .5).long()
        pred = self(img)
        loss = criterion(pred, mask)
        return loss

    
class RegressionModel(BaseModel):
    def step(self, batch, criterion):
        img, mask = batch["image"], batch["target"]
        pred = self(img)
        loss = criterion(pred, mask)
        return loss

class ClassificationModel(BaseModel):
    def step(self, batch, criterion):
        img, label = batch["image"], batch["label"]
        pred = self(img)
        loss = criterion(pred, label)
        return loss
    
"""
Datasets
"""

class BaseDataset(Dataset):
    def __len__(self):
        return len(self.df) * self.fold
    
    def __getitem__(self, idx):
        return self.aug(**self.df.iloc[idx % len(self.df)])
    
    def __repr__(self):
        sample = self[0]
        print("N =", len(self))
        for k, v in sample.items():
            print(k, v.dtype, v.shape, sep="\t")
        show(sample)
        return ""
    
    
class SegmentationDataset(BaseDataset):
    def __init__(self, df, aug=A.NoOp(), fold=1):
        """
        df.iloc[:, 0]: (C, H, W) float32
        df.iloc[:, 1]: (H, W)    float32
        """
        self.df = pd.DataFrame(dict(
            image=df.iloc[:, 0],
            mask=df.iloc[:, 1]
        ))
        self.aug = A.Compose([
            aug,
            ToTensorV2()
        ])
        self.fold = fold

    
class RegressionDataset(BaseDataset):
    def __init__(self, df, aug=A.NoOp(), fold=1):
        """
        df.iloc[:, 0]: (C, H, W) float32
        df.iloc[:, 1]: (C, H, W) float32
        """
        self.df = pd.DataFrame(dict(
            image=df.iloc[:, 0],
            target=df.iloc[:, 1]
        ))
        self.aug = A.Compose([
            aug,
            ToTensorV2()
        ], additional_targets=dict(
            target="image"
        ))
        self.fold = fold

class ClassificationDataset(BaseDataset):
    def __init__(self, df, aug=A.NoOp(), fold=1):
        """
        df.iloc[:, 0]: (C, H, W) float32
        df.iloc[:, 1]: long
        """
        self.df = pd.DataFrame(dict(
            image=df.iloc[:, 0],
            label=df.iloc[:, 1]
        ))
        self.aug = A.Compose([
            aug,
            ToTensorV2()
        ])
        self.fold = fold
        
class ClassificationOnMaskDataset(BaseDataset):
    def __init__(self, df, aug=A.NoOp(), fold=1):
        """
        df.iloc[:, 0]: (C, H, W) float32
        df.iloc[:, 1]: (H, W)    float32
        df.iloc[:, 2]: long
        """
        self.df = pd.DataFrame(dict(
            image=df.iloc[:, 0],
            mask=df.iloc[:, 1],
            label=df.iloc[:, 2]
        ))
        self.aug = A.Compose([
            aug,
            ToTensorV2()
        ])
        self.fold = fold
        
"""
Trainer
"""
    
class Trainer:
    def __init__(self, name, model, batch_size, train_set, val_set=None, device=None):
        self.name = name
        self.model = model
        self.batch_size = batch_size
        self.train_set = train_set
        self.val_set = val_set
        self.device = device
        
        self.best_model = None
        self.last_model = None
        
        plt.rcParams["figure.facecolor"] = "white"
    
    def __auto_select():
        output = subprocess.check_output([
    "nvidia-smi", "--format=csv", "--query-gpu=memory.used"
        ])
        df = pd.read_csv(io.BytesIO(output), names=["used_memory"], skiprows=1)

        df.used_memory = (
            df.used_memory
            .apply(lambda used_memory:
                   int(used_memory[:-4]))
        )
        return torch.device(f"cuda:{df.used_memory.idxmin()}")
    
    def fit(self, epochs, criterion):
        try:
            device = self.device if self.device else Trainer.__auto_select()
            self.model.to(device)
            optimizer = optim.Adam(self.model.parameters())

            train_loader = DataLoader(
                self.train_set,
                shuffle=True,
                batch_size=self.batch_size,
                num_workers=8
            )

            if self.val_set:
                val_loader = DataLoader(
                    self.val_set,
                    shuffle=False,
                    batch_size=self.batch_size,
                    num_workers=8
                )
                
            mb = master_bar(range(1, epochs + 1))
            if self.val_set:
                mb.names = ["valid", "train"]
            else:
                mb.names = ["train"]

            train_losses, val_losses = [], []
            for epoch in mb:
                train_loss, val_loss = 0, 0
                x = range(1, epoch+1)
                
                self.model.train()
                for batch in progress_bar(train_loader, parent=mb):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    loss = self.model.step(batch, criterion)
                    train_loss += loss.item()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    batch = loss = None
                train_loss /= len(self.train_set)
                train_losses.append(train_loss)
                
                if self.val_set:
                    self.model.eval()
                    with torch.no_grad():
                        for batch in progress_bar(val_loader, parent=mb):
                            batch = {k: v.to(device) for k, v in batch.items()}
                            loss = self.model.step(batch, criterion)
                            val_loss += loss.item()
                            batch = loss = None
                    val_loss /= len(self.val_set)
                    val_losses.append(val_loss)
                    
                    graphs = [[x, val_losses], [x, train_losses]]
                    y = np.concatenate((train_losses, val_losses))
                else:
                    graphs = [[x, train_losses]]
                    y = train_losses
                
                x_margin = 0.2
                y_margin = 0.05
                x_bounds = [1-x_margin, epoch+x_margin]
                y_bounds = [np.min(y)-y_margin, np.max(y)+y_margin]

                mb.update_graph(graphs, x_bounds, y_bounds)
                
                
                if val_loss <= min(val_losses):
                    self.best_model = f"models/{self.name}_{epoch:04d}.pth"
                    self.model.save(self.best_model)
                    print(self.best_model, val_loss)
            
            self.last_model = f"models/{self.name}_{epoch:04d}.pth"
            self.model.save(self.last_model)
            print()
            print("last_model:", self.last_model, val_loss)
            print("best_model:", self.best_model, min(val_losses))
                    
        except:
            traceback.print_exc()
        finally:
            batch = loss = optimizer = None
            self.model.cpu()
            gc.collect()
            torch.cuda.empty_cache()