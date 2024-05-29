from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

def lire_alpha_digit(char):
    """
    Read and return the binaryalphadigs.mat dataset for the given characters.

    Args:
        char (list): List of characters to learn.

    Returns:
        numpy.ndarray: The data as a numpy array.
    """
    mat = loadmat("data/binaryalphadigs.mat")
    data = mat['dat']

    output = []

    for i in char:
        for j in range(38):
            out_inter = []
            for k in range(20):
                for l in range(16):
                    out_inter += [data[i][j][k][l]]
            output += [out_inter]
    return np.array(output)

def lire_MNIST(nums):
    """
    Read and return the mnist_all.mat dataset for the given numbers.

    Args:
        nums (list): List of numbers to learn.

    Returns:
        tuple: A tuple containing 4 numpy arrays - the data and the labels, seperated in training and test sets.
    """
    seuil = 128
    mat = loadmat("data/mnist_all.mat")
    X_train = []
    X_test = []
    Y_train = []
    Y_test = []
    
    for num in nums:
        data_train = mat[f'train{num}']
        data_test = mat[f'test{num}']
        
        for ligne in data_train:
            ligne_output = [1 if pixel >= seuil else 0 for pixel in ligne]
            X_train.append(ligne_output)
            
        for ligne in data_test:
            ligne_output = [1 if pixel >= seuil else 0 for pixel in ligne]
            X_test.append(ligne_output)
            
        one_hot = [1 if i == num else 0 for i in nums]
        Y_train.extend([one_hot] * len(data_train))
        Y_test.extend([one_hot] * len(data_test))

    return np.array(X_train), np.array(X_test), np.array(Y_train), np.array(Y_test)

def display_image(images, height, width, save=False):
    """
    Display the given images.

    Args:
        images (numpy.ndarray): The images as a numpy array.
        height (int): Height of each image.
        width (int): Width of each image.
        save (bool): Whether to save the image as a file (default: False).
    """
    images = np.array(images)
    nb_images = images.shape[0]
    generated_images_array = images.reshape(nb_images, height, width)

    # Display the generated images
    rows = 1  # Number of rows in the display grid
    cols = nb_images  # Number of columns in the display grid

    plt.figure(figsize=(10, 6))

    for i in range(nb_images):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(generated_images_array[i], cmap='gray')  # Set cmap to 'gray' for black and white
        plt.axis('off')
    
    plt.tight_layout()
    if save:
        plt.savefig("generated_image.png")
    plt.show()
    
# Ajout de la fonction sigmoïde pour un calcul plus propre et réutilisable
def sigmoid(x):
    return 1 / (1 + np.exp(-x))