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
# plt.show()

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
# plt.show()


'''
Build the model
Building the neural network requires configuring layers of the model, then compiling it.

Setup the layers
The basic building block of a neural network is the layer. A layer extracts a representation
from the data fed into it. Hopefully, a series of connected layers results in a representation
that is meaningful for the problem at hand. Much of deep learning consists of chaining together simple layers.
Most layers, like "tf.keras.layers.Dense", have internal parameters which are adjusted during training.
'''

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

'''
This network has 3 layers:
- input (tf.keras.layers.Flatten)
  - this layer transforms the images from a 2D array of 28x28 pixels to a 1D array of 784
  - no parameters to learn, only reformats the data
- hidden (tf.keras.layers.Dense)
  - densely connected layer of 126 neurons (nodes)
  - each takes input from all 784 nodes in the previous layer, weighting it according to hidden parameters
- output (tf.keras.layers.Dense)
  - 10-node softmax layer
  - each node represents a class of clothing
  - weighs inputs according to learned parameters outputting the probability that the image belongs to a given class

Compile the model
Before the model is ready for training, it needs a few more settings. These are added during the model's compile step:
- loss function: measures how far the model's outputs are from desired output
- optimizer: adjusts inner model parameters to minimize loss
- metrics: monitors training and testing steps e.g. accuracy; the fraction of images which were correctly classified
'''

model.complie(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


