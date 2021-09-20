import torch
from lib.csflib import student, gauss
from lib.datasets import AirFoilDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


NUM_NEURONS = 10

class Encoder(torch.nn.Module):
    def __init__(self,input_dim,latent_dim):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.fc1 = torch.nn.Linear(input_dim, NUM_NEURONS)
        self.fc2 = torch.nn.Linear(NUM_NEURONS, latent_dim)

        # setup the non-linearity
        self.act = torch.nn.CELU(inplace=True)

    def forward(self, x):
        h = x.view(-1, self.input_dim)
        h = self.act(self.fc1(h))
        z = self.fc2(h)
        return z

class Decoder(torch.nn.Module):
    def __init__(self,output_dim,latent_dim):
        super(Decoder, self).__init__()
        self.input_dim = output_dim
        self.latent_dim = latent_dim

        self.fc1 = torch.nn.Linear(latent_dim, NUM_NEURONS)
        self.fc2 = torch.nn.Linear(NUM_NEURONS, output_dim)

        # setup the non-linearity
        self.act = torch.nn.CELU(inplace=True)

    def forward(self, z):
        h = z.view(-1, self.latent_dim)
        h = self.act(self.fc1(h))
        x_rec = self.fc2(h)
        return x_rec

class AutoEmbedder(object):
    def __init__(self,input_dim,latent_dim,num_clusters,CSF=student,output_dist="gauss"):
        self.encoder = Encoder(input_dim,latent_dim)
        self.decoder = Decoder(input_dim,latent_dim)
        self.num_clusters = num_clusters
        self.CSF = CSF
        self.centers = torch.randn((num_clusters,latent_dim),requires_grad=True)
        self.output_dist = output_dist
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.targetBank = None

    def embeddingLoss(self,z,idx=None):
        P = self.calculateDist(z)
        if self.targetBank is None:
            Q = self.calculateTargetDist(P)
        else:
            Q = self.targetBank[idx,:]
        return torch.mean(Q*torch.log(Q/P),dim=0).sum()

    def reconLoss(self,x,x_rec):
        if self.output_dist == "gauss":
            return torch.mean(torch.square(x-x_rec),dim=0).sum()

    def calculateDist(self,z):
        D = torch.linalg.vector_norm(z.unsqueeze(-1)-self.centers.t().unsqueeze(0),dim=1)
        P = self.CSF(D)
        return P/torch.sum(P,dim=-1,keepdim=True)

    def calculateTargetDist(self,p):
        f = torch.sum(p,dim=0,keepdim=True)
        Q = torch.square(p)/f
        return Q/torch.sum(Q,dim=-1,keepdim=True).detach()
    
    def init_centers(self, epoch_loader=None, Z=None):
        from sklearn.cluster import KMeans

        with torch.no_grad():
            if Z is None:
                Z = torch.Tensor()
                for x, _ in epoch_loader:
                    Z = torch.cat([Z,self.encoder(x)])

            kmeans = KMeans(n_clusters=self.num_clusters).fit(Z.numpy())
            self.centers.data = torch.tensor(kmeans.cluster_centers_)
            print('Cluster centers are assigned with K-Means.')

class FuzzyRegressionModel(AutoEmbedder):
    def __init__(self,input_dim,latent_dim,num_clusters,CSF=student,output_dist="gauss"):
        super().__init__(input_dim,latent_dim,num_clusters,CSF,output_dist)
        self.num_rules = self.num_clusters
        self.B = torch.randn((self.latent_dim+1,self.num_rules),requires_grad=True)

    def fuzzify(self,z):
        dimwise_distance = torch.abs(z.unsqueeze(-1)-self.centers.t().unsqueeze(0))
        return self.CSF(dimwise_distance)

    def rulebase(self,mu):
        firings = torch.prod(mu,dim=1)
        return firings/torch.sum(firings,dim=1,keepdim=True)

    def defuzzify(self,normalized_firings,z):
        ## z: data_size x num_dims
        ## firings: data_size x num_rules
        ## B: (num_dims+1) x num_rules
        y = torch.matmul(torch.cat((z,torch.ones(z.shape[0],1)),dim=1),self.B) ## y: data_size x num_rules
        return torch.sum(normalized_firings*y,dim=1)

    def findOptimalB(self,normalized_firings,z,y):
        with torch.no_grad():
            z_ext = torch.cat((z,torch.ones(z.shape[0],1)),dim=1)
            coeff_mat = normalized_firings.unsqueeze(-2)*z_ext.unsqueeze(-1)
            coeff_vec = coeff_mat.flatten(start_dim=1)
            B_vec = torch.linalg.lstsq(coeff_vec,y)
            return B_vec.solution.view(self.latent_dim+1,self.num_rules)

    def regressionLoss(self,y,y_pred):
        return torch.mean(torch.square(y-y_pred))

    def loss(self,x,x_rec,z,y,y_pred,idx=None):
        recon_loss = self.reconLoss(x,x_rec)
        embedding_loss = self.embeddingLoss(z,idx)
        regression_loss = self.regressionLoss(y,y_pred)
        return [recon_loss,embedding_loss,regression_loss]

class AutoFuzzifier(object):
    def __init__(self,args):
        
        #region Take the arguments
        self.args = args
        kw = args._get_kwargs()
        for arg in kw:
            setattr(self,arg[0],arg[1])
        #endregion

        self.frm = FuzzyRegressionModel(input_dim=self.input_dim, 
                                        latent_dim=self.latent_dim, 
                                        num_clusters=self.num_clusters,
                                        CSF=student)

    def train_model(self):

        writer = SummaryWriter()
        
        dset = AirFoilDataset()
        dataLoader = DataLoader(dset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        epochLoader = DataLoader(dset, batch_size=dset.__len__(), shuffle=False, drop_last=False)

        optim = torch.optim.Adam([{'params':self.frm.encoder.parameters()},
                                {'params':self.frm.decoder.parameters()},
                                {'params':self.frm.centers}],lr=self.learning_rate)

        epx = 0
        for ii in range(3):
            for epoch in range(self.epochs[ii]):
                epx += 1
                for x, y, idx in dataLoader:
                    z = self.frm.encoder(x)
                    x_rec = self.frm.decoder(z)

                    if self.dimwiseMF:
                        mu = self.frm.fuzzify(z)
                        normalized_firings = self.frm.rulebase(mu)
                    else:
                        normalized_firings = self.frm.calculateDist(z)
                    y_pred = self.frm.defuzzify(normalized_firings,z)

                    optim.zero_grad()
                    loss = sum([loss*coeff for loss,coeff in zip(self.frm.loss(x,x_rec,z,y,y_pred,idx),self.loss_coeffs)][:ii+1])

                    loss.backward()
                    optim.step()

                for x, y, idx in epochLoader:
                    with torch.no_grad():
                        z = self.frm.encoder(x)
                        x_rec = self.frm.decoder(z)

                        if self.dimwiseMF:
                            mu = self.frm.fuzzify(z)
                            normalized_firings = self.frm.rulebase(mu)
                        else:
                            normalized_firings = self.frm.calculateDist(z)
                        self.frm.B = self.frm.findOptimalB(normalized_firings,z,y)
                        y_pred = self.frm.defuzzify(normalized_firings,z)

                        loss_list = self.frm.loss(x,x_rec,z,y,y_pred,idx) 
                        print(f"Reconstruction={loss_list[0].item():.4f}  Clustering={loss_list[1].item():.4f}  Regression={loss_list[2].item():.4f}")

                        if (ii>0 or epoch==self.epochs[0]-1) and self.use_target_bank:
                            self.frm.targetBank = self.frm.calculateTargetDist(self.frm.calculateDist(z))
                            print("New target distribution has been assigned.")

                        writer.add_scalar('Loss/Reconstruction', loss_list[0], epx, new_style=True)
                        writer.add_scalar('Loss/Clustering', loss_list[1], epx, new_style=True)
                        writer.add_scalar('Loss/Regression', loss_list[2], epx, new_style=True)

                        if epoch%10==0:
                            for dim in range(z.shape[-1]): 
                                writer.add_histogram(f"Embeddings/{dim=}",z[:,dim],epx)

            print("XXXXXXXXXXXXXXXXXXXXXXXXXXXX")
            if ii == 0: self.frm.init_centers(Z=z)

        writer.close()            


    def test():
        pass

