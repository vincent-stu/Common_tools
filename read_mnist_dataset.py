import numpy as np
import os
from PIL import Image


def load_mnist(mnist_image_file, mnist_label_file):
    with open(mnist_image_file, 'rb') as f1:
        image_file = np.frombuffer(f1.read(), np.uint8, offset=16).reshape(-1, 28 * 28)
    with open(mnist_label_file, 'rb') as f2:
        label_file = np.frombuffer(f2.read(), np.uint8, offset=8)
    img = Image.fromarray(image_file[7].reshape(28, 28))  # First image in the training set.
    img.show()  # Show the image


if __name__ == '__main__':
    train_image_file = r'D:\mnist_datasets\train-images-idx3-ubyte\train-images.idx3-ubyte'
    train_label_file = r'D:\mnist_datasets\train-labels-idx1-ubyte\train-labels.idx1-ubyte'

    load_mnist(train_image_file, train_label_file)