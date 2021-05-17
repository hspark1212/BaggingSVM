import tensorflow as tf
from tensorflow.keras.layers import Dense


class Ann(tf.keras.Model):
    def __init__(self):
        super(Ann, self).__init__()

        self.d1 = Dense(10, activation=tf.keras.activations.elu, kernel_initializer="he_normal")
        self.d2 = Dense(10, activation=tf.keras.activations.elu, kernel_initializer="he_normal")
        self.d3 = Dense(10, activation=tf.keras.activations.elu, kernel_initializer="he_normal")
        self.d4 = Dense(10, activation=tf.keras.activations.elu, kernel_initializer="he_normal")
        self.final_layer = Dense(1, activation=tf.keras.activations.sigmoid)

    @tf.function
    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        x = self.final_layer(x)
        return x
