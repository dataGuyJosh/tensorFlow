import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# tf.logging.set_verbosity(tf.logging.ERROR)
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

for i, c in enumerate(celsius):
    print(f'{c} degrees Celsius = {fahrenheit[i]} degrees Fahrenheit')


'''
Supervised ML Terminology
- features: inputs to model i.e. temperature in Celsius
- labels: output we're trying to predict i.e. temperature in Farenheit
- example: an input/output pair used during training
           i.e. temperature in Celsius/Farenheit at a specific index [22,72]
'''

'''
Creating a model
- units=1 --> specifies the number of neurons in the layer
- input_shape=[1] --> specifies that the input to this layer is a single value
                      i.e. the shape is a 1D array with 1 member

'''
# you can create layers individually
# l0 = tf.keras.layers.Dense(units=1, input_shape = [1])
# or as part of the model definition
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

'''
Compiling a model
Before training, the model needs to be compiled by providingthe
- loss: loss function, a way of measuring how far off predictions are from desired outcome
- optimizer: optimizer function, a way of adjusting internal values in order to reduce loss
'''

model.compile(
    loss='mean_squared_error',
    # 0.1 here indicates the learning rate
    optimizer=tf.keras.optimizers.Adam(0.1)
)

'''
Fitting (Training) a model
During training, the model takes Celsius values, 
performs calculations using weights and outputs predicted Farenheit values. 
Since weights are initially random, initial output will not be close correct values.
The difference between predicting and target values is calculated using the loss function,
the optimizer function directs how the weights should be adjusted.

The cycle of calculate, compare, adjust is controlled by the fit method.
- inputs (features)
- outputs (targets)
- epochs: how many times this cycle should run
- verbose: logging output
'''

history = model.fit(celsius, fahrenheit, epochs=500, verbose=False)

'''
Displaying training stats
The fit method returns a history object, we can use this object to plot how the
loss of our model decreases after each training epoch.
A high loss means predicted values do not correspond well to target values.

We will use matplotlib to visualize the loss.
'''

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(history.history['loss'])
plt.show()

'''
Making Predictions
The model has been trained using with 3500 examples. (7 Celsius/Farenheit pairs * 500 epochs)
We can make predictions about "unknown" temperatures using the "predict" function.
'''

print(f'Predicted: {model.predict([100.0])}, Target: 212')


# Layer Weights
print(f'Layer Weights: {model.get_weights()}') 
'''
This should output the following:
Layer Weights: [array([[1.8210231]], dtype=float32), array([29.232615], dtype=float32)]
Notice how these weights match closely to the real conversion formula?
For a single neuron with a single input & output, the activation function is the described by
    y = m * x + c
or in our case:
    f = 1.8 * c + 32
'''

