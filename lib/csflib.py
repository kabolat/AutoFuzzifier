import torch

def student(d,alpha=1):
    return torch.pow(1+torch.square(d)/alpha,-(alpha+1)/2)

def gauss(d,sigma=1):
    return torch.exp(.5*(d/sigma).square())