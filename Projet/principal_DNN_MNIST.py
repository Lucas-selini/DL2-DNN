"""
This script tests the performance of a Deep Neural Network (DNN) with pretraining on the MNIST dataset.

The script performs the following steps:
1. Loads the MNIST dataset for the specified digit classes.
2. Splits the dataset into training and testing sets.
3. Defines the parameters for the DNN and the training process.
4. Trains a DNN with pretraining
5. Calculate the mistake rate.
"""

from utils.dnnv2 import DNN
from utils.utils import *
from sklearn.model_selection import train_test_split

def main():
    # Load the MNIST dataset for the specified digit classes
    nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    X_train, X_test, Y_train, Y_test = lire_MNIST(nums)
    
    # Define the parameters for the DNN and the training process
    p = X_train.shape[1]
    q = 100
    learning_rate = 0.1
    batch_size = 50
    nb_iter = 75
    n_epochs = 15
    layers = [p, q, q, q, len(nums)]
    X_train_split, _, Y_train_split, _ = train_test_split(X_train, Y_train, train_size=20000, random_state=42)
    
    # Train a DNN with pretraining
    dnn = DNN(layers)
    dnn.pretrain_DNN(X_train_split, learning_rate, batch_size, nb_iter)
    dnn.retropropagation(X_train_split, Y_train_split, learning_rate, n_epochs, batch_size)
    pretrained_train_mistake_rate = dnn.test_DNN(X_train, Y_train)
    pretrained_test_mistake_rate = dnn.test_DNN(X_test, Y_test)

    # Calculate the mistake rate
    print(f"Pretrained DNN training mistake rate on the train set: {pretrained_train_mistake_rate}")
    print(f"Pretrained DNN testing mistake rate on the test set: {pretrained_test_mistake_rate}")
    
if __name__ == "__main__":
    main()