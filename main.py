import os

from config import set_parser
from dataset import make_dataset
from train import train_pu, train
from plot import plot_origin_data, plot_model_prediction


def main():

    parser = set_parser()
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # 1. make dataset

    x_data, y_data = make_dataset(data_style=args.data_style,
                                  n_samples=args.n_samples,
                                  n_positives=args.n_positives)
    plot_origin_data(x_data, y_data)
    # 2. train model

    print("training simple ann start")
    model = train(x=x_data,
                  y=y_data,
                  num_epochs=args.num_epochs,
                  batch_size=args.batch_size)

    print("training bagging ANN start")
    list_models = train_pu(x=x_data,
                           y=y_data,
                           num_iters=args.num_iters,
                           num_epochs=args.num_epochs,
                           batch_size=args.batch_size,
                           bagging_size=args.bagging_size)

    # 3. plot
    plot_model_prediction(model, x_data, y_data)
    plot_model_prediction(list_models, x_data, y_data)


if __name__ == '__main__':
    main()
