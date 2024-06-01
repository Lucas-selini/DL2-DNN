"""
This script tests the performance of a Deep Neural Network (DNN) with pretraining on the MNIST dataset.

The script performs the following steps:
1. Loads the MNIST dataset for the specified digit classes.
2. Splits the dataset into training and testing sets.
3. Defines the parameters for the DNN and the training process.
4. Trains a DNN with pretraining
5. Calculate the mistake rate.
"""

from utils.dnnv3 import DNN
from utils.utils import *
from sklearn.model_selection import train_test_split

def main():
    # Load the MNIST dataset for the specified digit classes
    nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # X_train, X_test, Y_train, Y_test = lire_MNIST(nums)
    # save X_train , X_test, Y_train, Y_test
    # np.save('X_train.npy', X_train)
    # np.save('X_test.npy', X_test)
    # np.save('Y_train.npy', Y_train)
    # np.save('Y_test.npy', Y_test)
    # read X_train , X_test, Y_train, Y_test
    X_train = np.load('X_train.npy')
    X_test = np.load('X_test.npy')
    Y_train = np.load('Y_train.npy')
    Y_test = np.load('Y_test.npy')
    # optimal
    print('X_train shape:', X_train.shape)
    p = 784 # X_train.shape[1]
    q = 200 
    # q = 100
    # Define the parameters for the DNN and the training process
    n_layers = 2
    n_classes = len(nums)
    learning_rate = 0.1
    batch_size = 64
    nb_iter = 75
    n_epochs = 125
    layers = [p, q, q]
    X_train_split, _, Y_train_split, _ = train_test_split(X_train, Y_train, train_size=30000, random_state=42)
    
    # Train a DNN with pretraining
    dnn = DNN(p, q, n_layers, n_classes, layers)
    dnn.pretrain_DNN(X_train_split, learning_rate, batch_size, nb_iter)
    dnn.retropropagation(X_train_split, Y_train_split, learning_rate, n_epochs, batch_size)
    pretrained_train_mistake_rate = dnn.test_DNN(X_train, Y_train)
    pretrained_test_mistake_rate = dnn.test_DNN(X_test, Y_test)

    # Calculate the mistake rate
    print(f"Pretrained DNN training mistake rate on the train set: {pretrained_train_mistake_rate}")
    print(f"Pretrained DNN testing mistake rate on the test set: {pretrained_test_mistake_rate}")
    
if __name__ == "__main__":
    main()