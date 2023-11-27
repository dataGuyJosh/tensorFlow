# TensorFlow
import tensorflow as tf
import tensorflow_datasets as tfds

# Import TensorFlow Datasets
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# Helper libraries
import math
import numpy as np
import matplotlib.pyplot as plt

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True, data_dir='udacity/data')
train_dataset, test_dataset = dataset['train'], dataset['test']

class_names = metadata.features['label'].names

num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples

'''
Each greyscale pixel has an integer value in the range [0,255].
For the model to work, we need to normalize the range to [0,1].
'''


def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels


# map applis normalize to each element in the dataset
train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

# take the first image & remove the colour dimension by reshaping
for image, label in test_dataset.take(1):
    image = image.numpy().reshape((28, 28))

# plot first image
plt.figure()
plt.imshow(image, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
# plt.show()

# plot first 25 images
plt.figure(figsize=(10,10))
for i, (image, label) in enumerate(train_dataset.take(25)):
    image = image.numpy().reshape((28,28))
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel(class_names[label])
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

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


'''
Train model
Define the iteration behaviour for the training dataset.
- dataset.repeat(): repeat training forever, limited by epochs parameter described below
- dataset.shuffle(60000): randomize order
- dataset.batch(32): tells "model.fit" to use batches of 32 images/labels
  when updating model variables

Training is performed by calling model.fit method:
- feed training data to model using "train_dataset"
- model learns associations between labels and images
- "epoch=5" parameter limits training to 5 full iterations of the training dataset,
  so a total of 5 * 60000 = 3000000 examples
'''
BATCH_SIZE = 32
train_dataset = train_dataset.cache().repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.cache().batch(BATCH_SIZE)

model.fit(train_dataset, epochs=5, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))


# Evaluate accuracy: compare model performance on test dataset
test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples/32))
print('Accuracy on test dataset:', test_accuracy)