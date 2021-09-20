import argparse
from lib.model import AutoFuzzifier
import torch


def main(args):

    model = AutoFuzzifier(args)
    # model.inference_test()
    if args.train:
        model.train_model()
    else:
        model.test()


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train',              default=True,           type=bool)
    parser.add_argument('--use_target_bank',    default=True,           type=bool)
    parser.add_argument('--dimwiseMF',          default=False,          type=bool)
    parser.add_argument('--latent_dim',         default=5,              type=int)
    parser.add_argument('--num_clusters',       default=5,              type=int)
    parser.add_argument('--input_dim',          default=5,              type=int)
    parser.add_argument('--epochs',             default=[50, 50, 50],  type=list)
    parser.add_argument('--loss_coeffs',        default=[1, 1, 1],   type=list)
    parser.add_argument('--batch_size',         default=128,             type=int)
    parser.add_argument('--learning_rate',      default=5e-4,           type=float)
    parser.add_argument('--dataset',            default="airfoil",      type=str)
    args = parser.parse_args()

    main(args)