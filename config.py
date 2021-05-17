import argparse


def set_parser():
    # Parse the arguments
    parser = argparse.ArgumentParser(description="baggingANN for PU learning")
    # init
    parser.add_argument("-d", "--device", default="0")
    # 1. make dataset
    parser.add_argument("--data_style", default="circle", help="circles, moons, blobs")
    parser.add_argument("-n", "--n_samples", default=60000)
    parser.add_argument("-n_p", "--n_positives", default=3000)
    # 2. train model
    parser.add_argument("--num_iters", default=10, help="# of models")
    parser.add_argument("--num_epochs", default=10)
    parser.add_argument("--batch_size", default=100)
    parser.add_argument("--bagging_size", default=1, help="# of bootstrap (multiple of positive)")

    return parser
