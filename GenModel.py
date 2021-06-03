import random
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from enum import Enum

class actiFunction(Enum):
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
    v = np.random.normal(u, q)
    if(up and v < u):
        v = 2*u - v
    return round(v)

def normalLawCat(list, val, minsigma = 2):
    v = abs(np.random.normal())
    if(v > minsigma):
        return random.choice(list).value
    else:
        return val

def normalActivation(val = actiFunction.RELU, minsigma=2):
    #print(random.choice(actiFunction))
    ret = normalLawCat(list(actiFunction), val, minsigma)
    return ret

def newHiddenLayer(Hunits=64, Hactivation="relu"):
    return Dense(units=normaLawInt(Hunits, Hunits/10), activation=normalActivation(Hactivation))

def Model(layers=1, inputs = 64, outputs=1):
    model = Sequential()
    layers = max(normaLawInt(layers, up=True), 1)
    #inputs = max(normaLawInt(inputs, up=True), inputs)
    #, input_dim=(1,)
    model.add(Dense(units=64, activation='relu', input_dim=inputs))
    for i in range(layers):
        linputs = max(normaLawInt(inputs, inputs/2, up=False), 1)
        model.add(newHiddenLayer(linputs))
    model.add(Dense(units=outputs, activation=normalActivation('sigmoid')))
    model.compile(optimizer='sgd', loss='mean_squared_error', metrics=['mae'])
    #model.summary()
    return model