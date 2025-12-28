"""
Neural network model generation with stochastic architecture selection.

This module provides functions to generate neural network models with
randomly varied architectures using normal distributions. This approach
allows for exploring different model configurations during training.
"""

import random
import numpy as np
import keras
from enum import Enum


class actiFunction(Enum):
    """
    Enumeration of available activation functions for neural network layers.
    
    These are standard Keras activation functions that can be randomly
    selected during model generation.
    """
    ELU = "elu"
    SOFTMAX = "softmax"
    SELU = "selu"
    SOFTPLUS = "softplus"
    SOFTSIGN = "softsign"
    RELU = "relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    HARD_SIGMOID = "hard_sigmoid"
    EXPONENTIAL = "exponential"
    LINEAR = "linear"

def normaLawInt(u=0, q=1, up=False):
    """
    Generate an integer from a normal distribution.
    
    Args:
        u (float): Mean of the normal distribution (default: 0)
        q (float): Standard deviation of the normal distribution (default: 1)
        up (bool): If True and value < mean, reflect it above the mean (default: False)
    
    Returns:
        int: Rounded value from the normal distribution
    """
    v = np.random.normal(u, q)
    if(up and v < u):
        v = 2*u - v
    return round(v)

def normalLawCat(list, val, minsigma=2):
    """
    Randomly choose between a default value and a random list element.
    
    Uses a normal distribution to decide: if |N(0,1)| > minsigma,
    return a random element from the list; otherwise return the default value.
    
    Args:
        list (list): List of possible values to choose from
        val: Default value to return
        minsigma (float): Threshold for standard deviations (default: 2)
    
    Returns:
        The default value or a random element's value from the list
    """
    v = abs(np.random.normal())
    if(v > minsigma):
        return random.choice(list).value
    else:
        return val

def normalActivation(val=actiFunction.RELU, minsigma=2):
    """
    Select an activation function, potentially varying from default.
    
    Args:
        val (actiFunction or str): Default activation function (default: RELU)
        minsigma (float): Threshold for variation (default: 2)
    
    Returns:
        str: Name of the selected activation function
    """
    ret = normalLawCat(list(actiFunction), val, minsigma)
    return ret


def newHiddenLayer(Hunits=64, Hactivation="relu"):
    """
    Create a new hidden layer with stochastically varied parameters.
    
    The number of units is drawn from a normal distribution around Hunits,
    and the activation function may be randomly changed.
    
    Args:
        Hunits (int): Target number of units (default: 64)
        Hactivation (str): Default activation function (default: "relu")
    
    Returns:
        keras.layers.Dense: A configured dense layer
    """
    return keras.layers.Dense(
        units=normaLawInt(Hunits, Hunits/10), 
        activation=normalActivation(Hactivation)
    )


def Model(layers=1, inputs=64, outputs=1):
    """
    Generate a sequential neural network model with stochastic architecture.
    
    The model structure is varied using normal distributions:
    - Number of layers varies around the target
    - Number of units per layer varies
    - Activation functions may vary
    
    Args:
        layers (int): Target number of hidden layers (default: 1)
        inputs (int): Number of input features (default: 64)
        outputs (int): Number of output units (default: 1)
    
    Returns:
        keras.models.Sequential: A compiled neural network model
    """
    model = keras.models.Sequential()
    layers = max(normaLawInt(layers, up=True), 1)
    
    # Input layer
    model.add(keras.layers.Dense(units=64, activation='relu', input_dim=inputs))
    
    # Hidden layers with varied architecture
    for i in range(layers):
        linputs = max(normaLawInt(inputs, inputs/2, up=False), 1)
        model.add(newHiddenLayer(linputs))
    
    # Output layer
    model.add(keras.layers.Dense(units=outputs, activation=normalActivation('sigmoid')))
    
    # Compile model
    model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['mae'])
    
    return model