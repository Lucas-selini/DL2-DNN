from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

plt.use('TkAgg')

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
        tuple: A tuple containing two numpy arrays - the data and the labels.
    """
    seuil = 128
    mat = loadmat("data/mnist_all.mat")
    Y = []
    output = []
    for num in nums:
        data = mat[f'train{num}']
        for ligne in data:
            ligne_output = [1 if pixel >= seuil else 0 for pixel in ligne]
            output.append(ligne_output)
        one_hot = [1 if i == num else 0 for i in nums]
        Y.extend([one_hot] * len(data))

    return np.array(output), np.array(Y)

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
        plt.imshow(generated_images_array[i], cmap='gray')
        plt.axis('off')

    plt.tight_layout()
    if save:
        plt.savefig("generated_image.png")
    plt.show()