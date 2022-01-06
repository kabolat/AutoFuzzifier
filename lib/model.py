import torch
from lib.datasets import return_data, return_dim
from torch.utils.tensorboard import SummaryWriter


class AutoFuzzifier(object):
    def __init__(self,args):
        
        #region Local libs
        from lib.csflib import student, gauss
        from lib.submodels import FuzzyRegressionModel
        #endregion

        #region Take the arguments
        self.args = args
        kw = args._get_kwargs()
        for arg in kw:
            setattr(self,arg[0],arg[1])
        #endregion

        self.frm = FuzzyRegressionModel(input_dim=return_dim(self.dataset), 
                                        latent_dim=self.latent_dim, 
                                        num_clusters=self.num_clusters,
                                        CSF=student,
                                        fcm_loss=self.fcm_loss)

    def train_model(self):

        if self.viz:
            self.writer = SummaryWriter()
        
        train_loader, epoch_loader, valdata = return_data(self.dataset, self.batch_size, SVD=self.SVD)

        optim = torch.optim.Adam([{'params':self.frm.encoder.parameters()},
                                {'params':self.frm.decoder.parameters()},
                                {'params':self.frm.centers}],lr=self.learning_rate, weight_decay=0.01)

        loss_list_val = [0.0, 0.0, 0.0]
        epx = 0
        for ii in range(3):
            print("New loss term has been added.")
            for epoch in range(self.epochs[ii]):
                epx += 1
                for x, y, idx in train_loader:

                    z, x_rec, y_pred = self.frm.forward(x)

                    optim.zero_grad()
                    loss = sum([loss*coeff for loss,coeff in zip(self.frm.loss(x,x_rec,z,y,y_pred,idx),self.loss_coeffs)][:ii+1])
                    loss.backward()
                    optim.step()

                for x, y, _ in epoch_loader:
                    with torch.no_grad():
                        z, x_rec, y_pred = self.frm.forward(x,y)

                        # loss_list = self.frm.loss(x,x_rec,z,y,y_pred) 
                        # print(f"Reconstruction={loss_list[0].item():.4f}  Clustering={loss_list[1].item():.4f}  Regression={loss_list[2].item():.4f}")
                        
                        if epx%self.val_rate==0:
                            z, x_rec, y_pred = self.frm.forward(valdata[:,:-1])

                            loss_list_val = self.frm.loss(valdata[:,:-1],x_rec,z,valdata[:,-1],y_pred,idx=None) 
                            print(f"VALIDATION:   Reconstruction={loss_list_val[0].item():.4f}  Clustering={loss_list_val[1].item():.4f}  Regression={loss_list_val[2].item():.4f}")
                            

                        if (ii>0 or epoch==self.epochs[0]-1) and self.use_target_bank:
                            self.frm.targetBank = self.frm.calculateTargetDist(self.frm.calculateDist(z))
                            print("New target distribution has been assigned.")

                        #region Tensorboard Visualization
                        if self.viz:
                            # self.writer.add_scalar('Loss/Reconstruction', loss_list[0], epx)
                            self.writer.add_scalar('Loss/Reconstruction', loss_list_val[0], epx)
                            # self.writer.add_scalar('Loss/Clustering', loss_list[0], epx)
                            self.writer.add_scalar('Loss/Clustering', loss_list_val[1], epx)
                            # self.writer.add_scalar('Loss/Regression', loss_list[0], epx)
                            self.writer.add_scalar('Loss/Regression', loss_list_val[2], epx)

                            if epx%self.hist_rate==0:
                                for dim in range(z.shape[-1]): 
                                    self.writer.add_histogram(f"Embeddings/{dim=}",z[:,dim],epx)
                        #endregion

            if ii == 0: self.frm.init_centers(Z=z)


    def test_model(self):

        test_loader, dset = return_data(self.dataset, self.batch_size, train=False, SVD=self.SVD)
        
        with torch.no_grad():
            for x, y, idx in test_loader:
                z, x_rec, y_pred = self.frm.forward(x)
                loss_list = self.frm.loss(x,x_rec,z,y,y_pred,idx)
                print("Test Losses:")
                print(f"Reconstruction={loss_list[0].item():.4f}  Clustering={loss_list[1].item():.4f}  Regression={loss_list[2].item():.4f}")
                
                error = (y-y_pred)*dset.std[-1]
                rmse = error.square().mean().sqrt()
                print(f"Root Mean Squared Error: {rmse.item()}")
                print(f"Number of Parameters: {self.frm.num_parameters}")
            
            if self.viz:
                self.writer.add_hparams({'dataset':self.dataset,
                                         'model':self.model,
                                         'epochs':str(self.epochs),
                                         'loss_coeffs':str(self.loss_coeffs),
                                         'batch_size':int(self.batch_size),
                                         'lr':self.learning_rate,
                                         'SVD':self.SVD,
                                         'fcm_loss':self.fcm_loss,
                                         'latent_dim':int(self.latent_dim),
                                         'num_clusters':int(self.num_clusters),
                                         'num_params':self.frm.num_parameters,},
                                        {'hparam/RMSE':rmse})
            return rmse.item()


class VanillaRegression(object):
    def __init__(self,args):
        #region Local libs
        from lib.submodels import VanillaNN
        #endregion
        
        #region Take the arguments
        self.args = args
        kw = args._get_kwargs()
        for arg in kw:
            setattr(self,arg[0],arg[1])
        #endregion

        self.vrm = VanillaNN(input_dim=return_dim(self.dataset))

    def train_model(self):
        
        if self.viz:
            self.writer = SummaryWriter()

        train_loader, epoch_loader = return_data(self.dataset, self.batch_size, SVD=self.SVD)
        optim = torch.optim.Adam(self.vrm.parameters(),lr=self.learning_rate)

        for epoch in range(sum(self.epochs)):
            for x, y, _ in train_loader:
                y_pred = self.vrm.forward(x)

                optim.zero_grad()
                loss = torch.mean(torch.square(y-y_pred))
                loss.backward()
                optim.step()

            for x, y, _ in epoch_loader:
                with torch.no_grad():
                    y_pred = self.vrm.forward(x)
                    loss = torch.mean(torch.square(y-y_pred))
                    print(f"Regression={loss.item():.4f}")

                    if self.viz:
                        self.writer.add_scalar('Loss/Regression', loss, epoch, new_style=True)

    def test_model(self):

        test_loader, dset = return_data(self.dataset, self.batch_size, train=False, SVD=self.SVD)
        
        with torch.no_grad():
            for x, y, _ in test_loader:
                y_pred = self.vrm.forward(x)
                loss = torch.mean(torch.square(y-y_pred))
                print("Test Loss:")
                print(f"Regression={loss.item():.4f}")
                
                error = (y-y_pred)*dset.std[-1]
                rmse = error.square().mean().sqrt()
                print(f"Root Mean Squared Error: {rmse.item()}")
                print(f"Number of Parameters: {self.vrm._num_parameters()}")

            if self.viz:
                self.writer.add_hparams({'dataset':self.dataset,
                                         'model':self.model,
                                         'epochs':str(sum(self.epochs)),
                                         'batch_size':int(self.batch_size),
                                         'lr':self.learning_rate,
                                         'SVD':self.SVD,
                                         'latent_dim':int(self.latent_dim),
                                         'num_params':self.frm.num_parameters,},
                                        {'hparam/RMSE':rmse})