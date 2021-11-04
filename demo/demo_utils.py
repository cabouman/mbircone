import os,sys
import numpy as np
import matplotlib.pyplot as plt
import imageio
import urllib.request
import tarfile
import yaml
from PIL import Image

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


def image_resize(image, output_shape):
    """Resizes a 3D image volume by performing 2D resizing along the slices dimension

    Args:
        image (ndarray): 3D numpy array containing image volume with shape (slices, rows, cols)
        output_shape (tuple): (num_rows, num_cols) shape of resized output

    Returns:
        ndarray: 3D numpy array containing interpolated image with shape (num_slices, num_rows, num_cols).
    """

    image_resized = np.empty((image.shape[0],output_shape[0],output_shape[1]), dtype=image.dtype)
    for i in range(image.shape[0]):
        PIL_image = Image.fromarray(image[i])
        PIL_image_resized = PIL_image.resize((output_shape[1],output_shape[0]), resample=Image.BILINEAR)
        image_resized[i] = np.array(PIL_image_resized)

    return image_resized


def download_and_extract(download_url, save_dir):
    """ Given a download url, download the file from ``download_url`` , and save the file as ``save_dir``. 
        If the file already exists in ``save_dir``, user will be queried whether it is desired to download and overwrite the existing files.
        If the downloaded file is a tarball, then it will be extracted to ``save_dir``. 
    
    Args:
        download_url: An url to download the data. This url needs to be public.
        save_dir (string): Path to parent directory where downloaded file will be saved . 
    Return:
        string: path to downloaded file. This will be ``save_dir``+ downloaded_file_name 
            In case whereno download is performed, the function will return path to the existing local file.
            In case where a tarball file is downloaded and extracted, the function will return the path to the parent directory where the file is extracted to, which is the save as ``save_dir``. 
    """
    
    is_download = True 
    local_file_name = download_url.split('/')[-1]
    save_path = os.path.join(save_dir, local_file_name)
    if os.path.exists(save_path):
        is_download = query_yes_no(f"{save_path} already exists. Do you still want to download and overwrite the file?")
    if is_download:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # download the data from url.
        print("Downloading file ...")
        try:
            urllib.request.urlretrieve(download_url, save_path)
        except urllib.error.HTTPError as e:
            if e.code == 401:
                raise RuntimeError(f'HTTP status code {e.code}: URL authentication failed! Currently we do not support downloading data from a url that requires authentication.')
            elif e.code == 403:
                raise RuntimeError(f'HTTP status code {e.code}: URL forbidden! Please make sure the provided URL is public.')
            elif e.code == 404:
                raise RuntimeError(f'HTTP status code {e.code}: URL not Found! Please check and make sure the download URL provided is correct.')
            else:
                raise RuntimeError(f'HTTP status code {e.code}: {e.reason}. For more details please refer to https://en.wikipedia.org/wiki/List_of_HTTP_status_codes')
        except urllib.error.URLError as e:
            raise RuntimeError('URLError raised! Please check your internet connection.')
        print(f"Download successful! File saved to {save_path}")
    else:
        print("Skipped data download and extraction step.")
    # Extract the downloaded file if it is tarball
    if save_path.endswith(('.tar','.tar.gz')):
        if is_download:
            tar_file = tarfile.open(save_path)
            print("Extracting tarball file to {save_dir} ...")
            # Extract to save_dir.
            tar_file.extractall(save_dir)
            tar_file.close
            print(f"Extraction successful! File extracted to {save_dir}")
        save_path = save_dir
    # Parse extracted dir and extract data if necessary
    return save_path


def query_yes_no(question):
    """Ask a yes/no question via input() and return the answer.
        Code modified from reference: `https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input/3041990`
    
    Args:
        question (string): Question that is presented to the user.
    Returns:
        Boolean value: True for "yes" or "Enter", or False for "no".
    """
    
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    prompt = " [y/n, default=n] "
    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if choice == "":
            return False
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")
    return


def load_yaml(yml_path):
    """Load parameter from yaml configuration file.
    
    Args:
        yml_path (string): Path to yaml configuration file
    Returns:
        A dictionary with parameters for cluster.
    """
    
    with open(yml_path, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    return data_loaded
