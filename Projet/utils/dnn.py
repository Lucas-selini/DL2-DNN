from utils.dbn import DBN
import copy
import numpy as np
import matplotlib.pyplot as plt

class DNN():
    def __init__(self, couche):
        """
        Deep Neural Network (DNN) class.

        Args:
            couche (list): List of number of neurons for each layer.
        """
        self.dbn = DBN(couche)
        
    
    def pretrain_DNN(self, X, learning_rate, batch_size, nb_iter):
        """
        Pretrains the DNN using the Deep Belief Network (DBN).

        Args:
            X (np.array): Input data of size n*p.
            learning_rate (float): Learning rate.
            batch_size (int): Batch size.
            nb_iter (int): Number of iterations.
        """
        self.dbn.train(X, learning_rate, batch_size, nb_iter)
    
    def calcul_softmax(self, X):
        """
        Calculates the softmax function for the input array.

        Args:
            X (np.array): Input array of size n*q.

        Returns:
            (np.array): Array of size n*q.
        """
        return np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)

    def entree_sortie_reseau(self,X):
        """
        Performs the forward pass through the network.

        Args:
            X (np.array): Input data of size n*p.

        Returns:
            (list): List of size n_layers, each element is an array of size n*q.
            (np.array): Array of size n*q.
        """
        L = [X]
        for rbm in self.dbn.rbms:
            X = rbm.entree_sortie(X)
            L.append(X)
        return L, self.calcul_softmax(L[-1])
    
    def retropropagation(self, X, Y, learning_rate, n_epochs, batch_size):
        """
        Performs the backpropagation algorithm to train the DNN.

        Args:
            X (np.array): Input data of size n*p.
            Y (np.array): Target labels of size n*q.
            learning_rate (float): Learning rate.
            n_epochs (int): Number of epochs.
            batch_size (int): Batch size.
        """
        X_copy = X.copy()
        losses = []

        for epoch in range(n_epochs):
            # Iterate over batches
            for j in range(0, X_copy.shape[0], batch_size):
                X_batch = X_copy[j:min(j + batch_size, X_copy.shape[0])]
                Y_batch = Y[j:min(j + batch_size, X_copy.shape[0])]
                tb = X_batch.shape[0]

                # Forward pass
                L, Y_hat = self.entree_sortie_reseau(X_batch)

                # Calculate the error
                delta = Y_hat - Y_batch

                # Create a copy of the DBN
                dbn_copy = copy.deepcopy(self.dbn)

                # Backpropagation
                for i in range(self.dbn.n_layers - 2, -1, -1):
                    # Update biases
                    dbn_copy.rbms[i].b -= learning_rate * np.mean(delta, axis=0)

                    # Update weights
                    dbn_copy.rbms[i].W -= learning_rate * np.dot(L[i].T, delta) / tb

                    # Calculate the error for the previous layer
                    delta = np.dot(delta, self.dbn.rbms[i].W.T) * L[i] * (1 - L[i])

                # Update the DBN
                self.dbn = dbn_copy

            # Calculate the loss
            L, Y_hat = self.entree_sortie_reseau(X_copy)
            loss = -np.mean(Y * np.log(Y_hat))

            # Print the loss every 5 epochs
            if epoch % 5 == 0:
                print(f"Loss at epoch {epoch}: {loss}")

            # Store the loss
            losses.append(loss)
            
        # plt.plot(range(len(losses)),losses)
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.title('Loss over epochs')
        # plt.show()
        # plt.savefig('loss.png')
        # plt.close()

    def test_DNN(self, X, Y):
        """
        Tests the performance of the DNN on the given data.

        Args:
            X (np.array): Input data of size n*p.
            Y (np.array): Target labels of size n*q.

        Returns:
            (float): Accuracy of the DNN.
        """
        # Process the input through the network
        _, Y_hat = self.entree_sortie_reseau(X)

        # Compare the predicted and actual classes
        correct_predictions = np.argmax(Y_hat, axis=1) == np.argmax(Y, axis=1)

        # Calculate the accuracy
        accuracy = np.mean(correct_predictions)

        return accuracy