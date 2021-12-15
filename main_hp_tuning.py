import argparse

from optuna.study.study import ObjectiveFuncType
from lib.model import AutoFuzzifier, VanillaRegression
import optuna

class Args():
    pass

def main(trial):
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', choices=["fuzzy", "vanilla"],    default="fuzzy",type=str)
    parser.add_argument('--use_target_bank', default=False, type=bool)
    parser.add_argument('--fcm_loss', default=trial.suggest_categorical("fcm",[True,False]), type=bool)
    parser.add_argument('--dimwiseMF', default=False, type=bool)
    parser.add_argument('--SVD', default=True, type=bool)
    parser.add_argument('--latent_dim', default=trial.suggest_int("latent_dim", 1, 10), type=int)
    parser.add_argument('--num_clusters', default=trial.suggest_int("clusters", 2, 20, log=True), type=int)
    parser.add_argument('--epochs', nargs="*",
                        default=[
                            trial.suggest_int("epoch1", 100, 300, log=True), 
                            trial.suggest_int("epoch2", 100, 300, log=True), 
                            trial.suggest_int("epoch3", 100, 300, log=True)], type=int)
    parser.add_argument('--loss_coeffs', nargs="*", default=[
                            trial.suggest_loguniform("loss1", 0.1, 10), 
                            trial.suggest_loguniform("loss2", 0.1, 10), 
                            trial.suggest_loguniform("loss3", 0.1, 10)], type=float)
    parser.add_argument('--batch_size', default=trial.suggest_categorical("bs",[32,64,128,256]), type=int)
    parser.add_argument('--learning_rate', default=trial.suggest_float("lr", 1e-4, 1e-2, log=True), type=float)
    parser.add_argument('--dataset', default="airfoil", type=str)

    parser.add_argument('--viz', default=True, type=bool)
    parser.add_argument('--hist_rate', default=50, type=int)

    args = parser.parse_args()

    model = AutoFuzzifier(args)

    model.train_model()
    rmse = model.test_model()
    
    return rmse

if __name__=="__main__":
    
    study = optuna.create_study(direction="minimize")
    study.optimize(main, n_trials=200, timeout=None)

