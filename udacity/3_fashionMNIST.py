# TensorFlow
import tensorflow as tf
import tensorflow_datasets as tfds

# Helpers
import math
import numpy as np
import matplotlib.pyplot as plt

# improve progress bar display (???)
import tqdm
import tqdm.auto

tqdm.tdqm = tqdm.auto.tqdm

dataset, metadata = tfds.load(
    'fashion_mnist', as_supervised=True, with_info=True, data_dir='udacity/data')
trn, tst = dataset['train'], dataset['test']

class_names = ['T-shirt/top', 'Trousers', 'Pullover', 'Dress',
               'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


'''
Each greyscale pixel has an integer value in the range [0,255].
For the model to work, we need to normalize the range to [0,1].
'''


def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels


# map applis normalize to each element in the dataset
trn = trn.map(normalize)
tst = tst.map(normalize)

# take the first image & remove the colour dimension by reshaping
for image, label in tst.take(1):
    image = image.numpy().reshape((28, 28))

# plot first image
plt.figure()
plt.imshow(image, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()

# plot first 25 images
plt.figure(figsize=(10, 10))
i = 0
for img, label in tst.take(25):
    img = img.numpy().reshape((28, 28))
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img, cmap=plt.cm.binary)
    plt.xlabel(class_names[label])
    i += 1
plt.show()


