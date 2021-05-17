import matplotlib.pyplot as plt
from metrics import prediction_score


def plot_origin_data(x, y):
    # Plot the data set, as the models will see it
    plt.scatter(
        x[y == 0][:, 0], x[y == 0][:, 1],
        c='k', marker='.', linewidth=1, s=10, alpha=0.5,
        label='Unlabeled'
    )
    plt.scatter(
        x[y == 1][:, 0], x[y == 1][:, 1],
        c='b', marker='o', linewidth=0, s=50, alpha=0.5,
        label='Positive'
    )
    plt.legend()
    plt.title('Data set')

    plt.savefig("./figure/origin_data.png")


def plot_model_prediction(model, x, y):
    u = x[y == 0]
    if isinstance(model, list):
        list_models = model
        pred = prediction_score(list_models, u)
        title = "baggingANN"
    else:
        pred = model(u)
        title = "simpleANN"

    plt.scatter(
        u[:, 0], u[:, 1],
        c=pred, cmap="jet", marker='.', linewidth=1, s=10, alpha=0.5,
    )
    plt.colorbar(label='Scores given to unlabeled points')
    plt.title(title)
    plt.savefig(f"./figure/{title}.png")
    plt.show()
