import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd


class AirFoilDataset(Dataset):
    dim = 5
    
    def __init__(self, csv_file="datasets/airfoil/airfoil_self_noise.csv"):
        self.data = pd.read_csv(csv_file,sep='\t',header=None).values
        self.data = torch.tensor(self.data).float()
        
        self.mean = self.data.mean(dim=0)
        self.std = self.data.std(dim=0)
        self.data = (self.data-self.mean)/self.std


    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return self.data[idx,:-1], self.data[idx,-1], idx

def return_data(dset_name, batch_size, SVD=True):
    
    if dset_name == "airfoil":
        dset = AirFoilDataset()
    else:
        raise NotImplementedError
    
    if SVD:
        applySVD(dset)
    
    train_loader = DataLoader(dset, batch_size=batch_size, shuffle=True, drop_last=True)
    epoch_loader = DataLoader(dset, batch_size=dset.__len__(), shuffle=False, drop_last=False)

    return train_loader, epoch_loader

def return_dim(dset_name):
    if dset_name == "airfoil":
        return AirFoilDataset.dim
    else:
        raise NotImplementedError

def applySVD(dset):
    _, _, vh = torch.linalg.svd(dset.data, full_matrices=False)
    dset.data = torch.matmul(dset.data, vh.t())
