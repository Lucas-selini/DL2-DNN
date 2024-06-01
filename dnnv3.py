from utils.dbnv2 import DBN
from utils.rbmv2 import RBM  # Assuming RBM is defined similarly as in the initial script
from scipy.special import softmax
import numpy as np
import matplotlib.pyplot as plt

class DNN():
    def __init__(self, p, q, n_layers, n_classes, layers):
        """
        Deep Neural Network (DNN) class.

        Args:
            p (int): Number of input features.
            q (int): Number of hidden units.
            n_layers (int): Number of layers.
            n_classes (int): Number of output classes.
        """
        self.p_ = p
        self.q_ = q
        self.n_layers_ = n_layers
        self.n_classes_ = n_classes
        self.dbn = DBN(layers)
        self.classifier = RBM(q, n_classes)
        # print('self.classifier b shape:', self.classifier.b.shape)
        # print('self.classifier W shape:', self.classifier.W.shape)

    def pretrain_DNN(self, data, epochs, learning_rate, batch_size, verbose=False, plot=False):
        """
        Pretrains the DNN using the Deep Belief Network (DBN).

        Args:
            data (np.array): Input data of size n*p.
            epochs (int): Number of epochs.
            learning_rate (float): Learning rate.
            batch_size (int): Batch size.
            verbose (bool): If True, print training progress.
            plot (bool): If True, plot training loss.
        """
        self.dbn.train(data, epochs, learning_rate, batch_size, verbose, plot)
    
    # def calcul_softmax(self, X):
    #     """
    #     Calculates the softmax function for the input array.

    #     Args:
    #         X (np.array): Input array of size n*q.

    #     Returns:
    #         (np.array): Array of size n*q.
    #     """
    #     return np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)

    def calcul_softmax(self, data):
        prob = np.dot(data, self.classifier.W) + self.classifier.b
        return softmax(prob, axis=1)

    def entree_sortie_reseau(self, data):
        """
        Performs the forward pass through the network.

        Args:
            data (np.array): Input data of size n*p.

        Returns:
            (list): List of activations for each layer and the softmax output.
        """
        X = data.copy()
        sorties = [X]
        for rbm in self.dbn.rbms:
            X = rbm.entree_sortie(X)
            sorties.append(X)
            # print('entree sortie reseau - X shape:', X.shape)
        sorties.append(self.calcul_softmax(sorties[-1]))
        return sorties

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
            # permutation = np.random.permutation(X_copy.shape[0])
            # X_copy = X_copy[permutation]
            # Y_copy = Y[permutation]
            Y_copy = Y.copy()

            # Iterate over batches
            for j in range(0, X_copy.shape[0], batch_size):
                X_batch = X_copy[j:min(j + batch_size, X_copy.shape[0])]
                Y_batch = Y_copy[j:min(j + batch_size, X_copy.shape[0])]
                tb = X_batch.shape[0]

                # Forward pass
                # L, Y_hat = self.entree_sortie_reseau(X_batch)
                L = self.entree_sortie_reseau(X_batch)
                Y_hat = L[-1]  # M x n_classes

                # Calculate the errorsorties
                delta = Y_hat - Y_batch

                # print('delta shape:', delta.shape)

                # Update weights and biases for the classifier
                db = np.sum(delta, axis=0) / tb
                # print('db shape:', db.shape)
                dW = np.dot(L[-2].T, delta) / tb
                # print('classifier b shape:', self.classifier.b.shape)
                # print('classifier W shape:', self.classifier.W.shape)
                self.classifier.b -= learning_rate * db
                self.classifier.W -= learning_rate * dW

                # Backpropagation through hidden layers
                delta = np.dot(delta, self.classifier.W.T) * L[-2] * (1 - L[-2])

                for i in reversed(range(len(self.dbn.rbms))):
                    rbm = self.dbn.rbms[i]
                    db = np.sum(delta, axis=0) / tb
                    dW = np.dot(L[i].T, delta) / tb
                    rbm.b -= learning_rate * db
                    rbm.W -= learning_rate * dW
                    if i != 0:
                        delta = np.dot(delta, rbm.W.T) * L[i] * (1 - L[i])

            # Calculate the loss
            L_ = self.entree_sortie_reseau(X_copy)
            Y_hat = L_[-1]
            loss = -np.mean(Y * np.log(Y_hat))

            # Print the loss every 10% of epochs
            if verbose and epoch % (n_epochs // 10) == 0:
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
        print("Erreur :", 1 - accuracy)

        return 1 - accuracy
