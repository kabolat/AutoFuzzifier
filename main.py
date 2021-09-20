import argparse
from lib.model import AutoFuzzifier, VanillaRegression

def main(args):

    if args.model == "fuzzy":
        model = AutoFuzzifier(args)
    elif args.model == "vanilla":
        model = VanillaRegression(args)

    if args.train:
        model.train_model()
    else:
        model.test()


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', choices=["fuzzy", "vanilla"],    default="vanilla",      type=str)

    parser.add_argument('--train', default=True, type=bool)
    parser.add_argument('--use_target_bank', default=False, type=bool)
    parser.add_argument('--dimwiseMF', default=False, type=bool)
    parser.add_argument('--SVD', default=True, type=bool)
    parser.add_argument('--latent_dim', default=5, type=int)
    parser.add_argument('--num_clusters', default=5, type=int)
    parser.add_argument('--epochs', default=[50, 50, 50], type=list)
    parser.add_argument('--loss_coeffs', default=[1, 1, 1], type=list)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--learning_rate', default=5e-4, type=float)
    parser.add_argument('--dataset', default="airfoil", type=str)

    parser.add_argument('--viz', default=False, type=bool)
    parser.add_argument('--hist_rate', default=10, type=int)

    args = parser.parse_args()

    main(args)