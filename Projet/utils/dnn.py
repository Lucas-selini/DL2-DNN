from utils.dbn import DBN
import copy
import numpy as np
import matplotlib.pyplot as plt

class DNN():
    def __init__(self, couche):
        """
        Args:
            couche (list): list of number of neurons for each layer
        """
        self.dbn = DBN(couche)
        
    
    def pretrain_DNN(self, X, learning_rate, batch_size, nb_iter):
        """
        Args:
            X (np.array): size n*p
            nb_iter (int): number of iterations
            learning_rate (float): learning rate
            batch_size (int): batch size
        """
        self.dbn.train(X, learning_rate, batch_size, nb_iter)
    
    def calcul_softmax(self, X):
        """
        Args:
            X (np.array): size n*q
        Return:
            (np.array) array of size n*q
        """
        return np.exp(X) / np.sum(np.exp(X), axis=1, keepdims=True)

    def entree_sortie_reseau(self,X):
        """
        Args:
            X (np.array): size n*p
        Return:
            (list) list of size n_layers, each element is an array of size n*q
            (np.array) array of size n*q
        """
        L = [X]
        for rbm in self.dbn.rbms:
            X = rbm.entree_sortie(X)
            L.append(X)
        return L, self.calcul_softmax(L[-1])
    
    def retropropagation(self, X, Y, learning_rate,n_epochs,batch_size):
        """
        Args:
            X (np.array): size n*p
            Y (np.array): size n*q
            learning_rate (float): learning rate
            n_epochs (int): number of epochs
            batch_size (int): batch size
        """
        X_copy= X.copy()
        losses = []

        for epoch in range(n_epochs):
            for j in range(0, X_copy.shape[0], batch_size):
                X_batch = X_copy[j:min(j+batch_size, X_copy.shape[0])]
                Y_batch = Y[j:min(j+batch_size, X_copy.shape[0])]
                tb = X_batch.shape[0]
                L, Y_hat = self.entree_sortie_reseau(X_batch)
                delta = Y_hat - Y_batch
                dbn_copy = copy.deepcopy(self.dbn)
                
                for i in range(self.dbn.n_layers-2, -1, -1):
                    dbn_copy.rbms[i].b -= learning_rate * np.mean(delta, axis=0)
                    dbn_copy.rbms[i].W -= learning_rate * np.dot(L[i].T, delta) / tb
                    delta = np.dot(delta, self.dbn.rbms[i].W.T) * L[i] * (1 - L[i])
                    
                self.dbn = dbn_copy

            L, Y_hat = self.entree_sortie_reseau(X_copy)
            loss = -np.mean(Y * np.log(Y_hat))
            print(f"Loss at epoch {epoch} : {loss}")
            losses.append(loss)
        plt.plot(range(len(losses)),losses)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss over epochs')
        plt.show()

    def test_DNN(self, X, Y):
        """
        Args:
            X (np.array): size n*p
            Y (np.array): size n*q
        Return:
            (float) accuracy
        """
        # Process the input through the network
        _, Y_hat = self.entree_sortie_reseau(X)

        # Compare the predicted and actual classes
        correct_predictions = np.argmax(Y_hat, axis=1) == np.argmax(Y, axis=1)

        # Calculate the accuracy
        accuracy = np.mean(correct_predictions)

        return accuracy