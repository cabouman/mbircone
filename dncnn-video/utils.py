import os
import sys
from glob import glob
import numpy as np
from PIL import Image
import array
import pdb
from ruamel.yaml import YAML
from skimage.filters import gaussian
import time

#############################################################################
## Semi 2D
#############################################################################

def semi2DCNN_inPad(size_z_in, size_z_out):
    padSize_in1 = (size_z_in-size_z_out)//2
    padSize_in2 = size_z_in-size_z_out-padSize_in1
    return padSize_in1, padSize_in2

def semi2DCNN_select_z_out_from_z_in(img_patch, size_z_in, size_z_out):
    padSize_in1, padSize_in2 = semi2DCNN_inPad(size_z_in, size_z_out)
    # range does not work with tf
    return img_patch[:,:,:,padSize_in1:padSize_in1+size_z_out]

#############################################################################
## Generic Helper Routines
#############################################################################



def load_images(filelist):
    # pixel value range 0-255
    if not isinstance(filelist, list):
        im = Image.open(filelist).convert('L')
        return np.array(im).reshape(1, im.size[1], im.size[0], 1)
    data = []
    for file in filelist:
        im = Image.open(file).convert('L')
        data.append(np.array(im).reshape(1, im.size[1], im.size[0], 1))
    return data

def load_images_norm(filelist, normVal=255.0):
    # pixel value range 0-255
    if not isinstance(filelist, list):
        im = Image.open(filelist).convert('L')
        im = np.array(im, dtype='f').reshape(1, im.size[1], im.size[0], 1)/normVal
        return im
    data = []
    for file in filelist:
        im = Image.open(file).convert('L')
        im = np.array(im, dtype='f').reshape(1, im.size[1], im.size[0], 1)/normVal
        data.append(im)
    return data


def load_images_cont_norm(filelist, normVal=255.0):
    # pixel value range 0-255
    if not isinstance(filelist, list):
        im = Image.open(filelist).convert('L')
        im = np.array(im, dtype='f').reshape(1, im.size[1], im.size[0], 1)/normVal
        return im

    im = Image.open(filelist[0]).convert('L')
    size_vect = [len(filelist), im.size[1], im.size[0], 1]
    data = np.zeros( size_vect, dtype='f')
    for idx in range(len(filelist)):
        file = filelist[idx]
        im = Image.open(file).convert('L')
        im = np.array(im, dtype='f').reshape(1, im.size[1], im.size[0], 1)/normVal
        data[idx] = im
    # print(data.shape)
    return data


def save_images(filepath, ground_truth, noisy_image=None, clean_image=None):
    # assert the pixel value range is 0-255
    ground_truth = np.squeeze(ground_truth)
    noisy_image = np.squeeze(noisy_image)
    clean_image = np.squeeze(clean_image)
    if not clean_image.any():
        cat_image = ground_truth
    else:
        cat_image = np.concatenate([ground_truth, noisy_image, clean_image], axis=1)
    im = Image.fromarray(cat_image.astype('uint8')).convert('L')
    im.save(filepath, 'png')

def save_images_norm(filepath, scaleVal, clipVal, img1, img2=None, img3=None):
    # assert the pixel value range is 0-255
    img1 = np.squeeze(img1)
    img2 = np.squeeze(img2)
    img3 = np.squeeze(img3)
    if not img3.any():
        cat_image = img1
    else:
        cat_image = np.concatenate([img1, img2, img3], axis=1)

    cat_image_q = np.clip(scaleVal * cat_image, 0, clipVal).astype('uint8')
    im = Image.fromarray(cat_image_q).convert('L')
    im.save(filepath, 'png')


def save_summary(inputPatches_arr, folderName='./patch_summary', img_root='patch', num_patch_summary=100, verbose=1):

    patch_count = inputPatches_arr.shape[0]
    num_patch_summary = min(num_patch_summary, patch_count)

    if verbose==1:
        print(folderName)
    if not os.path.exists(folderName):
        os.makedirs(folderName)

    patchID_list = np.linspace(0, patch_count-1, num_patch_summary, dtype=int).tolist()
    for i, patch_id in enumerate(patchID_list):

        patch_3D = inputPatches_arr[patch_id,:,:,:]

        for index_2D in range(inputPatches_arr.shape[3]):
            patch_2D = inputPatches_arr[patch_id,:,:,index_2D].astype(np.float32)
            # print(patch_2D.shape)
            img_name = img_root + '_%d_%d.png'%(i,index_2D)
            filename_summary = os.path.join(folderName, img_name)
            
            if patch_2D.shape[0]!=1 and patch_2D.shape[1]!=1:
                # only save summary for 2D images
                save_images_norm(filename_summary, 255.0, 255.0, patch_2D)

        if verbose==1:
            print('Patch limits: {} , {}'.format(patch_3D.min(),patch_3D.max()) )
            # print(patch_3D.shape)
            # print("patch_id=%d rmse=%f, 99perc = %f" %( patch_id, np.sqrt(np.mean(patch_2D**2)), np.percentile(patch_2D, 99) )  )
            # print(filename_summary)


def makeDirInPath(path):
    dirName = os.path.dirname(path)
    if not os.path.exists(dirName):
        os.makedirs(dirName)


#############################################################################
## Metrics
#############################################################################

def cal_psnr(im1, im2, maxval=1.0):
    # assert pixel value range is 0-255 and type is uint8
    mse = ((im1.astype(np.float) - im2.astype(np.float)) ** 2).mean()
    psnr = 10 * np.log10(maxval ** 2 / mse)
    return psnr, mse



#############################################################################
## Load/Save Binary Files
#############################################################################


def read_stackOfFileNames(dirPathList, file_ext):

    # each directory must have stack of recon files (4D)
    stackOfFileNames = []
    for dirPath in dirPathList:

        wildCardFname = os.path.join(dirPath,'*.'+file_ext)
        fNameList = glob(wildCardFname)
        fNameList.sort()
        stackOfFileNames.append(fNameList)

    return stackOfFileNames


def write4D(x, fNameList, isRecon=True):

    assert len(fNameList)==x.shape[0], 'write4D: dimension mismatch'

    for i, fName in enumerate(fNameList):
        temp = x[i]
        write3D(temp, fName, isRecon)


def read4D(fNameList, isRecon=True):

    if len(fNameList)==0 :
        x = np.asarray([])
    else:
        x = read3D(fNameList[0], isRecon)
        x = np.expand_dims(x,0)
        x = np.repeat(x, len(fNameList), 0)

        for i, fName in enumerate(fNameList):
            temp = read3D(fName, isRecon)
            x[i] = temp

    return x

def write3Dlist(x, fNameList, isRecon=True):

    assert len(fNameList)==len(x), 'write3Dlist: dimension mismatch'

    for i, fName in enumerate(fNameList):
        write3D(x[i], fName, isRecon)


def read3Dlist(fNameList, isRecon=True):

    x = []
    for fName in fNameList:
        temp = read3D(fName, isRecon)
        x.append(temp)

    return x


def write3D(x, filePath, isRecon=True):

    fileID = open(filePath, "wb")

    if isRecon:
        x = np.copy(np.swapaxes(x, 0, 2), order='C') # copy with C order

    N1 = x.shape[0]
    N2 = x.shape[1]
    N3 = x.shape[2]
    numElements = N1*N2*N3

    sizesArray = array.array('i')
    sizesArray.fromlist([N1, N2, N3])
    sizesArray.tofile(fileID)

    valuesArray = array.array('f')
    valuesArray.fromlist(x.astype('float32').flatten('C').tolist())
    valuesArray.tofile(fileID)


def read3D(filePath, isRecon=True):
    ''' 
    fileID = open(filePath, 'rb')
    print("currently reading: ", filePath)
    sizesArray = array.array('i') # int array
    sizesArray.fromfile(fileID, 3)
    N1 = sizesArray[0] # slowest index
    N2 = sizesArray[1]
    N3 = sizesArray[2]
    
    numElements = N1*N2*N3

    valuesArray = array.array('f') # float array
    valuesArray.fromfile(fileID, numElements)

    x = np.asarray(valuesArray).astype('float32').reshape([N1, N2, N3])
    '''
    x = np.load(filePath).astype('float32')
    print("size of loaded data before swapping axis = ", np.shape(x))
    print("min max of data = ", x.min(), x.max()) 
    if isRecon:
        x = np.copy(np.swapaxes(x, 0, 2), order='C') # copy with C order
    print("size of loaded data = ", np.shape(x))
    return x

#############################################################################
## Load/Save Image Lists
#############################################################################

def generateFileList(numFiles, fileRoot, suffix):

    fileList = []
    for i in range(numFiles):
        fileList.append(fileRoot+str(i)+suffix)

    return fileList


def writeFileList(filePath, fileList):

    fileID = open(filePath,'w')
    
    numFiles = len(fileList) 
    fileID.write(str(numFiles)+"\n")

    for fileName in fileList:
        fileID.write(fileName+"\n")


def readFileList(filePath, resolvePath=True):

    fileID = open(filePath,'r')

    lines = fileID.read().split("\n")
    fileID.close()

    numLines = int(lines[0])
    fileList = lines[1:numLines+1]

    if resolvePath==True:
        origDir = os.path.dirname(filePath)
        for i in range(len(fileList)):
            tempPath = os.path.join(origDir, fileList[i])
            fileList[i] = os.path.abspath( tempPath )

    return fileList


def get_params(params_path):

    with open(params_path, 'r') as fileID:
        yaml = YAML()
        params = yaml.load(fileID)

    for key,val in params['paths'].items():
        if not os.path.isabs(val):
            params['paths'][key] = os.path.realpath(os.path.join(os.path.dirname(params_path),val))

    return pythonize_params(params)


def print_params(params, start_str=''):

    for key,value in params.items():
        if isinstance(value, dict):
            print('{}:'.format(key))
            print_params(value, start_str='    ')
        else:
            print(start_str+'{}: {}'.format(key,value))


def pythonize_params(params):

    if isinstance(params, dict):
        params = dict(params)
        for key in params:
            params[key] = pythonize_params(params[key])

    if isinstance(params, float):
            params = float(params)

    if isinstance(params, int):
        params = int(params)

    if isinstance(params, str):
        params = str(params)

    if isinstance(params, list):
        params = list(params)
        for i, items in enumerate(params):
            if isinstance(items, float):
                params[i] = float(items)

    return params


def calc_upper_range(img, percentile=75, gauss_sigma=2., is_segment=True, threshold=0.5):
    '''
    Given a 4D image volume of shape (Nt, Nz, Nx, Ny), calculate the image upper range.
    '''
    assert(np.ndim(img)<=4), 'Error! Input image dim must be 4 or lower!' 
    if np.ndim(img) < 4:
        print(f"{np.ndim(img)} dim image provided. Automatically adding axes to make the input a 4D image volume.")    
        for _ in range(4-np.ndim(img)):
            img = np.expand_dims(img, axis=0)
    num_tp, num_axial_slices, _, _ = np.shape(img)
    start_time = time.time()
    img_smooth = np.array([[gaussian(img[t,i,:,:], gauss_sigma, preserve_range=True) for t in range(num_tp)] for i in range(num_axial_slices)])
    end_time = time.time()
    time_elapsed = end_time - start_time
    print(f"Gaussian filtering takes {time_elapsed:.2f} sec for an image of size ", img.shape)
    if is_segment:
        img_mean = np.mean(img_smooth)
        indicator = img_smooth > threshold * img_mean
    else:
        indicator = np.ones(img.shape)
    img_upper_range = np.percentile(img_smooth[indicator], percentile)
    #return img_upper_range, np.squeeze(img_smooth), np.squeeze(indicator)
    return img_upper_range
