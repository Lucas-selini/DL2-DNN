from utils.dbn import DBN
from utils.rbm import RBM
import numpy as np
import matplotlib.pyplot as plt

class DNN():
    def __init__(self, couche, n_classes):
        """
        Deep Neural Network (DNN) class.

        Args:
            couche (list): List of number of neurons for each layer.
            n_classes (int): Number of output classes.
        """
        self.dbn = DBN(couche)
        self.output_layer = RBM(couche[-1], n_classes)

    def pretrain_DNN(self, X, learning_rate, batch_size, nb_iter, verbose=False, plot=False):
        """
        Pretrains the DNN using the Deep Belief Network (DBN).

        Args:
            X (np.array): Input data of size n*p.
            learning_rate (float): Learning rate.
            batch_size (int): Batch size.
            nb_iter (int): Number of iterations.
            verbose (bool): If True, print training progress.
            plot (bool): If True, plot training loss.
        """
        self.dbn.train(X, learning_rate, batch_size, nb_iter, verbose, plot)

    def calcul_softmax(self, X):
        """
        Calculates the softmax function for the input array.

        Args:
            X (np.array): Input array of size n*q.

        Returns:
            (np.array): Array of size n*q.
        """
        prob = np.dot(X, self.output_layer.W) + self.output_layer.b
        return np.exp(prob) / np.sum(np.exp(prob), axis=1, keepdims=True)

    def entree_sortie_reseau(self, X):
        """
        Performs the forward pass through the network.

        Args:
            X (np.array): Input data of size n*p.

        Returns:
            (list): List of size n_layers, each element is an array of size n*q.
        """
        X_copy = X.copy()
        L = [X_copy]
        for rbm in self.dbn.rbms:
            X_copy = rbm.entree_sortie(X_copy)
            L.append(X_copy)
        L.append(self.calcul_softmax(L[-1]))
        return L

    def retropropagation(self, X, Y, learning_rate, n_epochs, batch_size, verbose=False, plot=True):
        """
        Performs the backpropagation algorithm to train the DNN.

        Args:
            X (np.array): Input data of size n*p.
            Y (np.array): Target labels of size n*q.
            learning_rate (float): Learning rate.
            n_epochs (int): Number of epochs.
            batch_size (int): Batch size.
            verbose (bool): If True, print loss at each 10% of epochs.
            plot (bool): If True, plot the loss curve.
        """
        X_copy = X.copy()
        losses = []

        for epoch in range(n_epochs):
            Y_copy = Y.copy()
            # Iterate over batches
            for j in range(0, X_copy.shape[0], batch_size):
                X_batch = X_copy[j:min(j + batch_size, X_copy.shape[0])]
                Y_batch = Y_copy[j:min(j + batch_size, X_copy.shape[0])]
                tb = X_batch.shape[0]

                # Forward pass
                L = self.entree_sortie_reseau(X_batch)
                Y_hat = L[-1]

                # Calculate the error
                delta = Y_hat - Y_batch

                # Update weights and biases for the output layer first
                db = np.sum(delta, axis=0) / tb
                dW = np.dot(L[-2].T, delta) / tb
                self.output_layer.b -= learning_rate * db
                self.output_layer.W -= learning_rate * dW

                # Backpropagation
                delta = np.dot(delta, self.output_layer.W.T) * L[-2] * (1 - L[-2])
                for i in reversed(range(len(self.dbn.rbms))):
                    rbm = self.dbn.rbms[i]
                    # Update biases
                    rbm.b -= learning_rate * np.sum(delta, axis=0) / tb
                    
                    # Update weights
                    rbm.W -= learning_rate * np.dot(L[i].T, delta) / tb
                    
                    # Calculate the error for the previous layer
                    if i != 0:
                        delta = np.dot(delta, rbm.W.T) * L[i] * (1 - L[i])

            # Calculate the loss
            L = self.entree_sortie_reseau(X_copy)
            Y_hat = L[-1]
            loss = -np.mean(Y * np.log(Y_hat))

            # Print the loss every 5 epochs
            if verbose and epoch % 5 == 0:
                print(f"Loss at epoch {epoch}: {loss}")

            # Store the loss
            losses.append(loss)

        # Plot the loss curve
        if plot:
            plt.plot(range(len(losses)), losses)
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Loss over epochs')
            plt.show()

    def test_DNN(self, X, Y):
        """
        Tests the performance of the DNN on the given data.

        Args:
            X (np.array): Input data of size n*p.
            Y (np.array): Target labels of size n*q.

        Returns:
            (float): Mistake rate of the DNN.
        """
        # Process the input through the network
        L_ = self.entree_sortie_reseau(X)
        Y_hat = L_[-1]
        
        # Compare the predicted and actual classes
        correct_predictions = np.argmax(Y_hat, axis=1) == np.argmax(Y, axis=1)

        # Calculate the accuracy
        accuracy = np.mean(correct_predictions)

        return 1 - accuracy
