# TensorFlow
import logging
import matplotlib.pyplot as plt
import numpy as np
import math
import tensorflow as tf
import tensorflow_datasets as tfds

# Import TensorFlow Datasets
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# Helper libraries
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

dataset, metadata = tfds.load(
    'fashion_mnist', as_supervised=True, with_info=True, data_dir='udacity/data')
train_dataset, test_dataset = dataset['train'], dataset['test']

class_names = metadata.features['label'].names

num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples

'''
The value of each pixel in the image data is an integer in the range [0,255].
For the model to work properly, these values need to be normalized to the range [0,1].
So here we create a normalization function, and then apply it to each image in the test and train datasets.
'''
def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels


# map applies normalize to each element in the dataset
train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

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
train_dataset = train_dataset \
    .cache() \
    .repeat() \
    .shuffle(num_train_examples) \
    .batch(BATCH_SIZE)

test_dataset = test_dataset.cache().batch(BATCH_SIZE)

model.fit(
    train_dataset,
    epochs=5,
    steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE)
)


# Evaluate accuracy: compare model performance on test dataset
test_loss, test_accuracy = model.evaluate(
    test_dataset, steps=math.ceil(num_test_examples/32))
print('Accuracy on test dataset:', test_accuracy)


# make predictions about some images
for test_images, test_labels in test_dataset.take(1):
    test_images = test_images.numpy()
    test_labels = test_labels.numpy()
    predictions = model.predict(test_images)


# graphing functions
def plot_image(i, predictions_array, true_labels, images):
    predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img[..., 0], cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 5
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)

plt.show()
