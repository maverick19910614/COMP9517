import numpy as np
import pandas as pd
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D

# Load data(do not change)
data = pd.read_csv("src/mnist_train.csv")
train_data = data[:2000]
test_data = data[2000:2500]

# ----- Prepare Data ----- #
# preparing your data including data normalization

# ----- Build CNN Network ----- #
# Define your model here
model = None

# ----- Define your loss function, optimizer and metrics ----- #
model.compile(None)

# ----- Complete PlotLearningCurve function ----- #
def PlotLearningCurve(epoch, trainingloss, testingloss):
    pass

# fit your model by using training data and get predict label
history = model.fit(None)

# plot learning curve
PlotLearningCurve(pass)

# evaluation your model by using testing data
score = model.evaluate(None)
print("final test accuracy", score)
