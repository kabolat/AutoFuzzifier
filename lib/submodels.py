import torch
from lib.csflib import student, gauss


NUM_NEURONS = 10
ACTIVATION = torch.nn.CELU(inplace=True)

class Encoder(torch.nn.Module):
    def __init__(self,input_dim,latent_dim):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.fc1 = torch.nn.Linear(input_dim, NUM_NEURONS)
        self.fc2 = torch.nn.Linear(NUM_NEURONS, NUM_NEURONS)
        self.fc3 = torch.nn.Linear(NUM_NEURONS, latent_dim)

        # setup the non-linearity
        self.act = ACTIVATION

    def forward(self, x):
        h = x.view(-1, self.input_dim)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        z = self.fc3(h)
        return z

    def _num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class Decoder(torch.nn.Module):
    def __init__(self,output_dim,latent_dim):
        super(Decoder, self).__init__()
        self.input_dim = output_dim
        self.latent_dim = latent_dim

        self.fc1 = torch.nn.Linear(latent_dim, NUM_NEURONS)
        self.fc2 = torch.nn.Linear(NUM_NEURONS, NUM_NEURONS)
        self.fc3 = torch.nn.Linear(NUM_NEURONS, output_dim)

        # setup the non-linearity
        self.act = ACTIVATION

    def forward(self, z):
        h = z.view(-1, self.latent_dim)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        x_rec = self.fc3(h)
        return x_rec

    def _num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class VanillaNN(torch.nn.Module):
    def __init__(self,input_dim):
        super(VanillaNN, self).__init__()
        self.input_dim = input_dim

        self.fc1 = torch.nn.Linear(input_dim, NUM_NEURONS)
        self.fc2 = torch.nn.Linear(NUM_NEURONS, NUM_NEURONS)
        self.fc3 = torch.nn.Linear(NUM_NEURONS, NUM_NEURONS)
        self.fc4 = torch.nn.Linear(NUM_NEURONS, 1)

        # setup the non-linearity
        self.act = ACTIVATION

    def forward(self, z):
        h = z.view(-1, self.input_dim)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        h = self.act(self.fc3(h))
        y = self.fc4(h).squeeze()
        return y

    def _num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class AutoEmbedder(object):
    def __init__(self,input_dim,latent_dim,num_clusters,CSF=student,output_dist="gauss", fcm_loss=False):
        self.encoder = Encoder(input_dim,latent_dim)
        self.decoder = Decoder(input_dim,latent_dim)
        self.num_clusters = num_clusters
        self.CSF = CSF
        self.centers = torch.randn((num_clusters,latent_dim),requires_grad=True)
        self.output_dist = output_dist
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.fcm_loss = fcm_loss
        self.targetBank = None
        
        self.num_parameters = self.encoder._num_parameters()+self.decoder._num_parameters()+self.centers.numel()

    def embeddingLoss(self,z,idx=None):
        if self.fcm_loss:
            return self.fcmLoss(z,idx)
        else:
            P = self.calculateDist(z)
            if self.targetBank is None:
                Q = self.calculateTargetDist(P)
            else:
                Q = self.targetBank[idx,:]
            return torch.sum(Q*torch.log(Q/P),dim=1).mean()
    
    def fcmLoss(self,z,idx):
        P = self.calculateDist(z)
        return torch.sum(torch.pow(P,2)*torch.linalg.vector_norm(z.unsqueeze(-1)-self.centers.t().unsqueeze(0),dim=1)**2,dim=1).mean()

    def reconLoss(self,x,x_rec):
        if self.output_dist == "gauss":
            return torch.sum(torch.square(x-x_rec),dim=1).mean()

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
    def __init__(self,input_dim,latent_dim,num_clusters,CSF=student,output_dist="gauss", dimwiseMF=False, fcm_loss=False):
        super().__init__(input_dim,latent_dim,num_clusters,CSF,output_dist,fcm_loss)
        self.num_rules = self.num_clusters
        self.dimwiseMF = dimwiseMF
        self.B = torch.randn((self.latent_dim+1,self.num_rules),requires_grad=True)
        self.num_parameters += self.B.numel()

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
    
    def forward(self,x):
        z = self.encoder(x)
        x_rec = self.decoder(z)

        if self.dimwiseMF:
            mu = self.frm.fuzzify(z)
            normalized_firings = self.rulebase(mu)
        else:
            normalized_firings = self.calculateDist(z)
        y_pred = self.defuzzify(normalized_firings,z)
        
        return z, x_rec, y_pred