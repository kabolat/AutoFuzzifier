import argparse
from lib.model import AutoFuzzifier, VanillaRegression

def main(args):

    if args.model == "fuzzy":
        model = AutoFuzzifier(args)
    elif args.model == "vanilla":
        model = VanillaRegression(args)

    model.train_model()
    model.test_model()

if __name__=="__main__":
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
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--dataset', default="airfoil", type=str)

    parser.add_argument('--viz', default=True, type=bool)
    parser.add_argument('--hist_rate', default=50, type=int)

    args = parser.parse_args()

    main(args)
    