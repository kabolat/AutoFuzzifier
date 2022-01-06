import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

class ModifiedDataset(Dataset):
    
    def prepareData(self,data_folder,train, SVD=True):
        if train:
            csv_file = data_folder+"/trainset.csv"
            self.data = pd.read_csv(csv_file,sep=',',header=None).values
            self.data = torch.tensor(self.data).float()
    
            self.mean = self.data.mean(dim=0)
            self.std = self.data.std(dim=0)
            self.data = (self.data-self.mean)/self.std
            
            if SVD: self.data[:,:-1], self.vh = self.applySVD(self.data[:,:-1])
            else: self.vh = None
            
            transform_dict = {"mean": self.mean, "std": self.std, "vh": self.vh}
            torch.save(transform_dict, data_folder+"/transform_dict.pt")
            
            csv_file = data_folder+"/valset.csv"
            self.valdata = pd.read_csv(csv_file,sep=',',header=None).values
            self.valdata = torch.tensor(self.valdata).float()
            
            self.valdata = (self.valdata-self.mean)/self.std
            if SVD: self.valdata[:,:-1] = torch.matmul(self.valdata[:,:-1], self.vh.t())
            
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
        
    def applySVD(self,data):
        _, _, vh = torch.linalg.svd(data, full_matrices=False)
        return torch.matmul(data, vh.t()), vh

    def splitDataset(self,data_folder, file_name, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
        from sklearn.model_selection import train_test_split
        
        df = pd.read_csv(data_folder+file_name, sep=',', header=None)
        train_val, test = train_test_split(df, train_size=train_ratio)
        train, val = train_test_split(train_val, train_size=self._relative_val_ratio(val_ratio,train_ratio))
        train.to_csv(data_folder+"/trainset.csv", index=False, header=False)
        val.to_csv(data_folder+"/valset.csv", index=False, header=False)
        test.to_csv(data_folder+"/testset.csv", index=False, header=False)

    def oneHotEncode(self,data_folder, file_name, category_columns):
        df = pd.read_csv(data_folder+file_name, sep=',', header=None)
        df = pd.concat([pd.get_dummies(df.iloc[:,category_columns]),df.drop(category_columns,axis=1)],axis=1)
        df
        df.to_csv(data_folder+"/one_hot.csv", index=False, header=False)

    @staticmethod
    def _relative_val_ratio(val_ratio, train_ratio):
        return val_ratio*(train_ratio+val_ratio)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return self.data[idx,:-1], self.data[idx,-1], idx


class AirFoilDataset(ModifiedDataset):
    dim = 5
    
    def __init__(self, data_folder="datasets/airfoil", train=True, SVD=True):
        
        if not os.path.isfile(data_folder+"/valset.csv"):
            print("The dataset required splitting!")
            file_name = "/airfoil_self_noise.csv"
            self.splitDataset(data_folder, file_name)
        self.prepareData(data_folder,train,SVD)

class AbaloneDataset(ModifiedDataset):
    dim = 10
    
    def __init__(self, data_folder="datasets/abalone", train=True, SVD=True):
        
        if not os.path.isfile(data_folder+"/valset.csv"):
            print("The dataset required splitting!")
            file_name = "/abalone.csv"
            self.oneHotEncode(data_folder,file_name,category_columns=0)
            self.splitDataset(data_folder, "/one_hot.csv")
        self.prepareData(data_folder,train,SVD)

class AutoMPGDataset(ModifiedDataset):
    dim = 9
    
    def __init__(self, data_folder="datasets/autompg", train=True, SVD=True):
        
        if not os.path.isfile(data_folder+"/valset.csv"):
            print("The dataset required splitting!")
            file_name = "/auto-mpg.csv"
            self.oneHotEncode(data_folder,file_name,category_columns=0)
            self.splitDataset(data_folder, "/one_hot.csv")
        self.prepareData(data_folder,train,SVD)

class ConcreteDataset(ModifiedDataset):
    dim = 8
    
    def __init__(self, data_folder="datasets/concrete", train=True, SVD=True):
        
        if not os.path.isfile(data_folder+"/valset.csv"):
            print("The dataset required splitting!")
            file_name = "/concrete.csv"
            self.splitDataset(data_folder, file_name)
        self.prepareData(data_folder,train,SVD)

class ProteinDataset(ModifiedDataset):
    dim = 9
    
    def __init__(self, data_folder="datasets/protein", train=True, SVD=True):
        
        if not os.path.isfile(data_folder+"/valset.csv"):
            print("The dataset required splitting!")
            file_name = "/protein.csv"
            self.splitDataset(data_folder, file_name)
        self.prepareData(data_folder,train,SVD)

class PowerPlantDataset(ModifiedDataset):
    dim = 4
    
    def __init__(self, data_folder="datasets/powerplant", train=True, SVD=True):
        
        if not os.path.isfile(data_folder+"/valset.csv"):
            print("The dataset required splitting!")
            file_name = "/powerplant.csv"
            self.splitDataset(data_folder, file_name)
        self.prepareData(data_folder,train,SVD)

class RedWineDataset(ModifiedDataset):
    dim = 11
    
    def __init__(self, data_folder="datasets/redwine", train=True, SVD=True):
        
        if not os.path.isfile(data_folder+"/valset.csv"):
            print("The dataset required splitting!")
            file_name = "/redwine.csv"
            self.splitDataset(data_folder, file_name)
        self.prepareData(data_folder,train,SVD)

class WhiteWineDataset(ModifiedDataset):
    dim = 11
    
    def __init__(self, data_folder="datasets/whitewine", train=True, SVD=True):
        
        if not os.path.isfile(data_folder+"/valset.csv"):
            print("The dataset required splitting!")
            file_name = "/whitewine.csv"
            self.splitDataset(data_folder, file_name)
        self.prepareData(data_folder,train,SVD)


def return_data(dset_name, batch_size, train=True, SVD=True):
    
    if dset_name == "airfoil":
        dset = AirFoilDataset(train=train, SVD=SVD)
    elif dset_name == "abalone":
        dset = AbaloneDataset(train=train, SVD=SVD)
    elif dset_name == "autompg":
        dset = AutoMPGDataset(train=train, SVD=SVD)    
    elif dset_name == "concrete":
        dset = ConcreteDataset(train=train, SVD=SVD)
    elif dset_name == "protein":
        dset = ProteinDataset(train=train, SVD=SVD)
    elif dset_name == "powerplant":
        dset = PowerPlantDataset(train=train, SVD=SVD)
    elif dset_name == "redwine":
        dset = RedWineDataset(train=train, SVD=SVD)
    elif dset_name == "whitewine":
        dset = WhiteWineDataset(train=train, SVD=SVD)
    else:
        raise NotImplementedError
    
    if train:
        train_loader = DataLoader(dset, batch_size=batch_size, shuffle=True, drop_last=True)
        epoch_loader = DataLoader(dset, batch_size=dset.__len__(), shuffle=False, drop_last=False)
        return train_loader, epoch_loader, dset.valdata, dset.std
    else:
        return DataLoader(dset, batch_size=dset.__len__(), shuffle=False, drop_last=False), dset
    

def return_dim(dset_name):
    if dset_name == "airfoil":
        return AirFoilDataset.dim
    elif dset_name == "abalone":
        return AbaloneDataset.dim
    elif dset_name == "autompg":
        return AutoMPGDataset.dim
    elif dset_name == "concrete":
        return ConcreteDataset.dim
    elif dset_name == "protein":
        return ProteinDataset.dim
    elif dset_name == "powerplant":
        return PowerPlantDataset.dim
    elif dset_name == "redwine":
        return RedWineDataset.dim
    elif dset_name == "whitewine":
        return WhiteWineDataset.dim
    else:
        raise NotImplementedError