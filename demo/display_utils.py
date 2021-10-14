
import numpy as np
import matplotlib.pyplot as plt
import imageio


def font_setting():
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 12
    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


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

    return rmse / denominator


def plt_cmp_3dobj(phantom, recon, display_slice=None, display_x=None, display_y=None, vmin=None, vmax=None,
                  filename=None):
    Nz, Nx, Ny = recon.shape
    if display_slice is None:
        display_slice = Nz // 2
    if display_x is None:
        display_x = Nx // 2
    if display_y is None:
        display_y = Ny // 2

    # Compute Normalized Root Mean Squared Error
    nrmse_1 = nrmse(recon, phantom)

    font_setting()

    # display phantom
    fig, axs = plt.subplots(2, 3)

    title = f'Phantom: Axial Scan {display_slice:d}.'
    axs[0, 0].imshow(phantom[display_slice], vmin=vmin, vmax=vmax, cmap='gray', interpolation='none')
    axs[0, 0].set_title(title)

    title = f'Phantom: Coronal Scan {display_x:d}.'
    axs[0, 1].imshow(phantom[:, display_x, :], vmin=vmin, vmax=vmax, cmap='gray', interpolation='none')
    axs[0, 1].set_title(title)

    title = f'Phantom: Sagittal Scan {display_y:d}.'
    axs[0, 2].imshow(phantom[:, :, display_y], vmin=vmin, vmax=vmax, cmap='gray', interpolation='none')
    axs[0, 2].set_title(title)

    # display reconstruction
    title = f'Recon: Axial Scan {display_slice:d}.'
    axs[1, 0].imshow(recon[display_slice], vmin=vmin, vmax=vmax, cmap='gray', interpolation='none')
    axs[1, 0].set_title(title)

    title = f'Recon: Coronal Scan {display_y:d}.'
    axs[1, 1].imshow(recon[:, display_x, :], vmin=vmin, vmax=vmax, cmap='gray', interpolation='none')
    axs[1, 1].set_title(title)

    title = f'Recon: Sagittal Scan {display_x:d}.'
    im = axs[1, 2].imshow(recon[:, :, display_y], vmin=vmin, vmax=vmax, cmap='gray', interpolation='none')
    axs[1, 2].set_title(title)

    # Add colorbar
    plt.subplots_adjust(wspace=0.9, hspace=0.4, right=0.8)
    cax = plt.axes([0.85, 0.15, 0.05, 0.7])
    plt.colorbar(im, cax=cax)

    plt.suptitle(f"3D Shepp-Logan Recon Phantom VS Recon with NRMSE={nrmse_1:.3f}.")

    if filename is not None:
        try:
            plt.savefig(filename, dpi=600)
        except:
            print("plot_image() Warning: Can't write to file {}".format(filename))


def plot_gif(x, save_dir, name, vmin=None, vmax=None):
    images = []
    for i in range(x.shape[0]):
        fig = plt.figure()
        plt.imshow(x[i], vmin=vmin, vmax=vmax, interpolation='none').set_cmap('gray')
        plt.title("Slice: %d" % (i + 1))
        plt.colorbar()
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)
        plt.close()
    imageio.mimsave(save_dir + "/%s.gif" % name, images, fps=5)



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
    imgplot = plt.imshow(img, vmin=vmin, vmax=vmax, interpolation='none')
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
        sizesArray = np.fromfile(fileID, dtype=ntype, count=n_dim)
        numElements = np.prod(sizesArray)
        dataArray = np.fromfile(fileID, dtype=dtype, count=numElements).reshape(sizesArray)

    return dataArray

