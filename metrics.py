import tensorflow as tf


@tf.function
def calculate_loss(y_true, y_pred):
    """
    :param y_true: tensor [B, ]
    :param y_pred: tensor [B, n]
    :return m : tf.float, binary crossentropy loss
    """
    loss = tf.keras.losses.binary_crossentropy(y_true[:, tf.newaxis], y_pred)
    return tf.reduce_mean(loss)


@tf.function
def calculate_acc(y_true, y_pred):
    """
    :param y_true: tensor [B, ]
    :param y_pred: tensor [B, n]
    :return m : float, bincary accuracy
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    m = tf.keras.metrics.binary_accuracy(y_true[:, tf.newaxis], y_pred)
    return len(tf.where(m == 1)) / len(m)


@tf.function
def calculate_precision(y_true, y_pred, eps=1e-8):
    """
    :param y_true: tensor [B, ]
    :param y_pred: tensor [B, ]
    :param eps: float, 1e-8
    :return: tf.float,  precision
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    tp = tf.multiply(y_true, y_pred)
    return tf.reduce_sum(tp) / (tf.reduce_sum(y_pred) + eps)


@tf.function
def calculate_recall(y_true, y_pred, eps=1e-8):
    """
    :param y_true: tensor [B, ]
    :param y_pred: tensor [B, ]
    :param eps: float, 1e-8
    :return: tf.float,  recall
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    tp = tf.multiply(y_true, y_pred)

    return tf.reduce_sum(tp) / (tf.reduce_sum(y_true) + eps)


def prediction_score(models, inputs, return_label=False):
    score = tf.zeros([inputs.shape[0], 1])
    for i in range(len(models)):
        m = models[i]
        score += m(inputs)

    ave_score = score / len(models)

    if return_label == True:
        return tf.squeeze(tf.where(ave_score >= 0.5, 1, 0))
    else:
        return ave_score
