import torch
from torch._C import dtype
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
import numpy as np


class AirFoilDataset(Dataset):
    def __init__(self, csv_file="datasets/airfoil/airfoil_self_noise.csv"):
        self.data = pd.read_csv(csv_file,sep='\t',header=None)
        self.mean = self.data.mean().values
        self.std = self.data.std().values
        self.data = (self.data-self.mean)/self.std

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return torch.tensor(self.data.iloc[idx,:-1]).float(), torch.tensor(self.data.iloc[idx,-1]).float(), idx


def return_data(self):
    name = self.dataset
    dset_dir = self.dset_dir
    batch_size = self.batch_size
    num_workers = 4

    dset = UCIDatasets(name.lower())
    train_data = dset.get_split()
    test_data = dset.get_split()

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    if test_data is not None:
        test_loader = DataLoader(test_data, batch_size=test_data.__len__(), shuffle=False, num_workers=num_workers)

    return train_loader, test_loader



class UCIDatasets():
    def __init__(self,  name,  data_path="", n_splits = 10):
        self.datasets = {
            "airfoil":"https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat"}
        self.data_path = data_path
        self.name = name
        self.n_splits = n_splits
        self._load_dataset()

    
    def _load_dataset(self):
        if self.name not in self.datasets:
            raise Exception("Not known dataset!")
        if not path.exists(self.data_path+"UCI"):
            os.mkdir(self.data_path+"UCI")

        url = self.datasets[self.name]
        file_name = url.split('/')[-1]
        if not path.exists(self.data_path+"UCI/" + file_name):
            urllib.request.urlretrieve(
                self.datasets[self.name], self.data_path+"UCI/" + file_name)
        data = None


        if self.name == "airfoil":
            data = pd.read_csv(self.data_path+'UCI/'+file_name,
                        header=0, delimiter="\s+").values
            self.data = data[np.random.permutation(np.arange(len(data)))]
        
        elif self.name == "yeast":
            data = pd.read_csv(self.data_path+'UCI/yeast.data',
                        header=0, delimiter="\s+").values
            self.data = data[np.random.permutation(np.arange(len(data)))]
        
        elif self.name == "abalone":
            data = pd.read_csv(self.data_path+'UCI/abalone.data',
                        header=0, delimiter=",").values
            data[data=="M"] = 0
            data[data=="I"] = 1
            data[data=="F"] = 2
            data = np.concatenate((data[:,1:], np.expand_dims(data[:,0],axis=-1)),axis=-1).astype('float64')
            self.data = data[np.random.permutation(np.arange(len(data)))]
        elif self.name == "iris":
            data = pd.read_csv(self.data_path+'UCI/iris.data',
                        header=0, delimiter=",").values
            data[data=="Iris-setosa"] = 0
            data[data=="Iris-versicolor"] = 1
            data[data=="Iris-virginica"] = 2
            self.data = data[np.random.permutation(np.arange(len(data)))].astype('float64')
        

        elif self.name == "concrete":
            data = pd.read_excel(self.data_path+'UCI/Concrete_Data.xls',
                                header=0).values
            self.data = data[np.random.permutation(np.arange(len(data)))]
        elif self.name == "energy":
            data = pd.read_excel(self.data_path+'UCI/ENB2012_data.xlsx',
                                header=0).values
            self.data = data[np.random.permutation(np.arange(len(data)))]
        elif self.name == "power":
            zipfile.ZipFile(self.data_path +"UCI/CCPP.zip").extractall(self.data_path +"UCI/CCPP/")
            data = pd.read_excel(self.data_path+'UCI/CCPP/Folds5x2_pp.xlsx', header=0).values
            np.random.shuffle(data)
            self.data = data
        elif self.name == "wine":
            data = pd.read_csv(self.data_path + 'UCI/winequality-red.csv',
                               header=1, delimiter=';').values
            self.data = data[np.random.permutation(np.arange(len(data)))]

        elif self.name == "yacht":
            data = pd.read_csv(self.data_path + 'UCI/yacht_hydrodynamics.data',
                               header=1, delimiter='\s+').values
            self.data = data[np.random.permutation(np.arange(len(data)))]
            
        self.in_dim = data.shape[1] - 1
        self.out_dim = 1

    def get_split(self, split=-1, train=True):
        x_train, y_train = self.data[:,:self.in_dim], self.data[:,self.in_dim:]
        self.num_class = np.unique(y_train).shape[0]
        x_means, x_stds = x_train.mean(axis=0), x_train.var(axis=0)**0.5
        x_train = (x_train - x_means)/x_stds
        inps = torch.from_numpy(x_train).float()
        tgts = torch.from_numpy(y_train).float()
        train_data = torch.utils.data.TensorDataset(inps, tgts)
        return train_data