# URLs
https://learn.udacity.com/courses/ud187


# Udacity TensorFlow Introduction
Artificial Intelligence: A field of computer science that aims to make computers achieve human-style intelligence. There are many approaches to reaching this goal, including machine learning and deep learning.

Machine Learning: A set of related techniques in which computers are trained to perform a particular task rather than by explicitly programming them.

Neural Network: A construct in Machine Learning inspired by the network of neurons (nerve cells) in the biological brain. Neural networks are a fundamental part of deep learning, and will be covered in this course.

Deep Learning: A subfield of machine learning that uses multi-layered neural networks. Often, “machine learning” and “deep learning” are used interchangeably.

Machine learning and deep learning also have many subfields, branches, and special techniques. A notable example of this diversity is the separation of Supervised Learning and Unsupervised Learning.

To over simplify — in supervised learning you know what you want to teach the computer, while unsupervised learning is about letting the computer figure out what can be learned. Supervised learning is the most common type of machine learning, and will be the focus of this course.


# Applications of ML
- skin cancer identification
- self-driving cars
- teaching computers to play games e.g. alpha go by google deep mind


# Dense Layers
- neurons in a dense layer are fully connected to neurons in adjacent layers
- less scalable than a sparse layer but much more "potent"

For a neural network with 3 inputs (x,y,z), 2 neurons in a hidden layer (a,b) and 1 neuron in the output layer (o).

w = weight --> w_xa = weight of x on a
b = bias --> ba = bias of b
a = x * w_xa + y * w_ya + z * w_za + ba
b = x * w_xb + y * w_yb + z * w_zb + bb
o = a * w_oa + b * w_ob + bo


# Classification vs Regression
- Regression: A model that outputs a single value. For example, an estimate of a house’s value.
- Classification: A model that outputs a probability distribution across several categories. For example, in Fashion MNIST, the output was 10 probabilities, one for each of the different types of clothing. Remember, we use Softmax as the activation function in our last Dense layer to create this probability distribution.

|                                | Classification                      | Regression                                              |
|--------------------------------|-------------------------------------|---------------------------------------------------------|
| Output                         | probability distribution of classes | single number                                           |
| Example                        | Identifying Clothing                | Predicting temperature tomorrow given temperature today |
| Loss                           | Sparse categorical crossentropy     | Mean squared error                                      |
| Last Layer Activation Function | Softmax                             | None                                                    |


# Convolution Neural Networks (CNNs)
- better than dense neural networks at image classification
- two main concepts; convolutions & max pooling

## Convolutions
