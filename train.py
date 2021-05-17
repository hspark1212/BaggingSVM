from model import Ann

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Progbar

from silence_tensorflow import silence_tensorflow

from dataset import DataloaderBagging, Dataloader

from metrics import calculate_loss, calculate_acc


@tf.function
def train_one_step(model, x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = calculate_loss(y_true=y, y_pred=logits)

    grads = tape.gradient(loss, model.weights)

    return logits, loss, grads


def train(x, y, num_epochs, batch_size):
    silence_tensorflow()

    model = Ann()
    optimizer = Adam()

    dataloader = Dataloader(x, y, batch_size=batch_size)

    for epoch in range(num_epochs):
        print(f"epoch {epoch + 1}/{num_epochs}")
        pb_i = Progbar(len(dataloader), interval=0.5)

        for batch_inputs, batch_labels in dataloader:

            logits, loss, grads = train_one_step(model, batch_inputs, batch_labels)
            optimizer.apply_gradients(zip(grads, model.weights))

            acc = calculate_acc(batch_labels, model(batch_inputs))
            values = [("loss", loss), ("acc", acc)]
            pb_i.add(1, values=values)

    return model


def train_pu(x, y, num_iters, num_epochs, batch_size, bagging_size):
    silence_tensorflow()

    list_models = []
    list_optimizers = []

    for i in range(num_iters):

        list_models.append(Ann())
        list_optimizers.append(Adam())

    dataloader_bagging = DataloaderBagging(x=x, y=y, bagging_size=bagging_size)

    for i in range(num_iters):
        print(f"====model {i + 1}====")
        p, b = dataloader_bagging[i]
        samples = tf.concat([p, b], axis=0)
        labels = tf.concat([tf.ones(len(p)), tf.zeros(len(b))], axis=0)

        dataloader = Dataloader(samples, labels, batch_size=batch_size)

        # set model and optimizer
        model = list_models[i]
        optimizer = list_optimizers[i]

        for epoch in range(num_epochs):
            print(f"epoch {epoch + 1}/{num_epochs}")
            pb_i = Progbar(len(dataloader), interval=0.5)

            for batch_inputs, batch_labels in dataloader:

                logits, loss, grads = train_one_step(model, batch_inputs, batch_labels)
                optimizer.apply_gradients(zip(grads, model.weights))

                acc = calculate_acc(batch_labels, model(batch_inputs))
                values = [("loss", loss), ("acc", acc)]
                pb_i.add(1, values=values)

        """print("========")
        pred = prediction_score(list_models[:i+1], x, return_label=True)
        p = calculate_precision(y, pred)
        r = calculate_recall(y, pred)
        print(f"precision : {p:.4f}, recall : {r:.4f}")"""

    return list_models
