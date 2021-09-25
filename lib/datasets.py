import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os

class AirFoilDataset(Dataset):
    dim = 5
    
    def __init__(self, data_folder="datasets/airfoil", train=True, SVD=True):
        
        if not os.path.isfile(data_folder+"/trainset.csv"):
            print("The dataset required splitting!")
            file_name = "/airfoil_self_noise.csv"
            splitDataset(data_folder, file_name)
        
        if train:
            csv_file = data_folder+"/trainset.csv"
            self.data = pd.read_csv(csv_file,sep=',',header=None).values
            self.data = torch.tensor(self.data).float()
    
            self.mean = self.data.mean(dim=0)
            self.std = self.data.std(dim=0)
            self.data = (self.data-self.mean)/self.std
            
            if SVD: self.data[:,:-1], self.vh = applySVD(self.data[:,:-1])
            else: self.vh = None
            
            transform_dict = {"mean": self.mean, "std": self.std, "vh": self.vh}
            torch.save(transform_dict, data_folder+"/transform_dict.pt")
        else:
            csv_file = data_folder+"/testset.csv"
            self.data = pd.read_csv(csv_file,sep=',',header=None).values
            self.data = torch.tensor(self.data).float()
            
            tr_dict = torch.load(data_folder+"/transform_dict.pt")
            self.mean = tr_dict["mean"]
            self.std = tr_dict["std"]
            self.vh = tr_dict["vh"]
            
            self.data = (self.data-self.mean)/self.std
            if SVD: self.data[:,:-1] = torch.matmul(self.data[:,:-1], self.vh.t())

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return self.data[idx,:-1], self.data[idx,-1], idx

def return_data(dset_name, batch_size, train=True, SVD=True):
    
    if dset_name == "airfoil":
        dset = AirFoilDataset(train=train, SVD=SVD)
    else:
        raise NotImplementedError
    
    if train:
        train_loader = DataLoader(dset, batch_size=batch_size, shuffle=True, drop_last=True)
        epoch_loader = DataLoader(dset, batch_size=dset.__len__(), shuffle=False, drop_last=False)
        return train_loader, epoch_loader
    else:
        return DataLoader(dset, batch_size=dset.__len__(), shuffle=False, drop_last=False), dset
    

def return_dim(dset_name):
    if dset_name == "airfoil":
        return AirFoilDataset.dim
    else:
        raise NotImplementedError

def applySVD(data):
    _, _, vh = torch.linalg.svd(data, full_matrices=False)
    return torch.matmul(data, vh.t()), vh

def splitDataset(data_folder, file_name, train_ratio=0.8):
    from sklearn.model_selection import train_test_split
    
    df = pd.read_csv(data_folder+file_name, sep=',', header=None)
    train, test = train_test_split(df, train_size=train_ratio)
    train.to_csv(data_folder+"/trainset.csv", index=False, header=False)
    test.to_csv(data_folder+"/testset.csv", index=False, header=False)