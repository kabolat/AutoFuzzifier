import argparse

from optuna.study.study import ObjectiveFuncType
from lib.model import AutoFuzzifier, VanillaRegression
import optuna

class hp_tune_wrapper(AutoFuzzifier):
    def __init__(self, args):
        super().__init__(args)
        
    def objective(self,trial):
    
        self.fcm_loss = trial.suggest_categorical("fcm",[True,False])
        self.latent_dim = trial.suggest_int("latent_dim", 1, 10)
        self.num_clusters = trial.suggest_int("clusters", 2, 20, log=True)
        self.epochs =  [trial.suggest_int("epoch1", 100, 300, log=True), 
                        trial.suggest_int("epoch2", 100, 300, log=True), 
                        trial.suggest_int("epoch3", 100, 300, log=True)]
        self.loss_coeffs = [trial.suggest_loguniform("loss1", 0.1, 10), 
                            trial.suggest_loguniform("loss2", 0.1, 10), 
                            trial.suggest_loguniform("loss3", 0.1, 10)]
        self.batch_size = trial.suggest_categorical("bs",[32,64,128,256])
        self.learning_rate = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        return self.train_model(trial=trial)


def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', choices=["fuzzy", "vanilla"],    default="fuzzy",type=str)

    parser.add_argument('--train', default=True, type=bool)
    parser.add_argument('--use_target_bank', default=False, type=bool)
    parser.add_argument('--fcm_loss', default=True, type=bool)
    parser.add_argument('--dimwiseMF', default=False, type=bool)
    parser.add_argument('--SVD', default=True, type=bool)
    parser.add_argument('--latent_dim', default=5, type=int)
    parser.add_argument('--num_clusters', default=10, type=int)
    parser.add_argument('--epochs', nargs="*", default=[300, 300, 400], type=int)
    parser.add_argument('--loss_coeffs', nargs="*", default=[10, 0.1, 1], type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--learning_rate', default=1e-2, type=float)
    parser.add_argument('--dataset', default="airfoil", type=str)

    parser.add_argument('--viz', default=True, type=bool)
    parser.add_argument('--hist_rate', default=100, type=int)
    parser.add_argument('--val_rate', default=10, type=int)
    
    parser.add_argument('--hp_tuning', default=True, type=bool)
    
    args = parser.parse_args()
    
    if args.hp_tuning and args.model=="fuzzy":
        model = hp_tune_wrapper(args)
        study = optuna.create_study(direction="minimize")
        study.optimize(model.objective, n_trials=100, timeout=None)
    else:
        if args.model == "fuzzy":
            model = AutoFuzzifier(args)
        elif args.model == "vanilla":
            model = VanillaRegression(args)
        model.train_model()

    model.test_model()
    return

if __name__=="__main__":
    main()
