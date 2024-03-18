from sklearn.preprocessing import OneHotEncoder
import numpy as np
from utils.dnn import DNN
from utils.utils import *
import time
import copy
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

p = 784
q = 50
learning_rate = 0.2
batch_size = 5
nb_iter = 100
num = 3

nb_gibbs_iteration = 20
nb_image_generate = 1

generated_images = []
errors = []
training_time = []

data = lire_MNIST(num)
display_image([data[500]],28,28,save=True)


# # Data loading and preprocessing for MNIST from the ipynb DNN notebook

# # Load the MNIST dataset
# mnist = fetch_openml('mnist_784')

# # Extract the features and labels

# X = mnist.data
# y = mnist.target

# X_bw = np.where(X > 127, 1, 0)

# X_train, X_test, y_train, y_test = train_test_split(X_bw, y, test_size=0.2, random_state=42)
# encoder = OneHotEncoder(sparse=False)
# y_train = encoder.fit_transform(y_train.to_numpy().reshape(-1, 1))
# y_test = encoder.fit_transform(y_test.to_numpy().reshape(-1, 1))