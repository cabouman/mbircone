
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio


def plot_gif(x,save_dir,name,vmin=0,vmax=0.05):
    images=[]
    for i in range(x.shape[0]):
        fig = plt.figure()
        plt.imshow(x[i], vmin=vmin,  vmax=vmax).set_cmap('gray')
        plt.colorbar()
        fig.canvas.draw() 
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)
        plt.close()
    imageio.mimsave(save_dir + "/%s.gif"%name, images, fps=5)

def plot_image(img, title=None, filename=None, vmin=None, vmax=None):
    """
    Function to display and save a 2D array as an image.

    Args:
        img: 2D numpy array to display
        title: Title of plot image
        filename: A path to save plot image
        vmin: Value mapped to black
        vmax: Value mapped to white
    """

    plt.ion()
    fig = plt.figure()
    imgplot = plt.imshow(img, vmin=vmin,  vmax=vmax)
    plt.title(label=title)
    imgplot.set_cmap('gray')
    plt.colorbar()
    if filename != None:
        try:
            plt.savefig(filename)
        except:
            print("plot_image() Warning: Can't write to file {}".format(filename))


def read_ND(filePath, n_dim, dtype='float32', ntype='int32'):

    with open(filePath, 'rb') as fileID:

        sizesArray = np.fromfile( fileID, dtype=ntype, count=n_dim)
        numElements = np.prod(sizesArray)
        dataArray = np.fromfile(fileID, dtype=dtype, count=numElements).reshape(sizesArray)

    return dataArray



def nrmse(image, reference_image):
    """
    Compute the normalized root mean square error between image and reference_image.
    Args:
        image: Calculated image
        reference_image: Ground truth image
    Returns:
        Root mean square of (image - reference_image) divided by RMS of reference_image
    """
    rmse = np.sqrt(((image - reference_image) ** 2).mean())
    denominator = np.sqrt(((reference_image) ** 2).mean())

    return rmse/denominator
