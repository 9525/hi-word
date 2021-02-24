import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
print(tf.test.gpu_device_name())
print(tf.test.is_gpu_available())
print('TF Version = {0:s}'.format(tf.__version__))
from tensorflow import keras
#from tensorflow.keras.datasets import mnist
#(X_train, Y_train), (X_test, Y_test) = keras.datasets.fashion_mnist.load_data()
mnist_data = keras.datasets.fashion_mnist.load_data()
print ("packs loaded")

def mnist_test():
    X_train = mnist_data[0][0]
    Y_train = mnist_data[0][1]
    X_test = mnist_data[1][0]
    Y_test = mnist_data[1][1]
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)

    X_train = X_train.reshape((60000, 28, 28, 1))
    X_test = X_test.reshape((10000, 28, 28, 1))
    print(X_train.shape)
    print(X_test.shape)

    # 归一化
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same", activation="relu",
                               input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),

        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),

        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3),  padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same"),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(200, activation='relu'),
        tf.keras.layers.Dense(10, activation='sigmoid')
    ])
    model.compile(optimizer='sgd',  # 随机梯度下降和动量优化
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()  # 打印模型
    history = model.fit(X_train, Y_train, epochs=10)

    # 将history中的数据以图片表示出来
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.ylim(0, 1)
    plt.show()

   # model.evaluate(X_test, Y_test)

if __name__ == '__main__':
    mnist_test()
