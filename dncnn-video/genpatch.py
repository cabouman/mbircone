import os
import sys
import argparse
import random
import numpy as np
import pdb
# pdb.set_trace()

from ruamel.yaml import YAML

from utils import *

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--params_path', dest='params_path', default='./params.yml')

args, extra = arg_parser.parse_known_args()
args = vars(args)


def isPatchGood(patch, params):

    mean = np.mean(patch)
    toss = np.random.uniform(low=0.0, high=1.0)

    if(mean < params['acceptThreshold_lo']):        
        return (toss>params['rejectProb_lo'])

    elif(mean > params['acceptThreshold_hi']):
        return (toss>params['rejectProb_hi'])

    else:
        return (toss>params['rejectProb_mid'])


def getPatch_3D(img_3D, idRange1, idRange2, idRange3):
    ''' convert 3D full sized image of size (Nx,Ny,Nz) to a list of patches with size params['patch_sizes']
        img_3D: full-size 3D image.
        idRange1: list [idx1_low, idx1_hi, idx1_stride], Specifying the first index of the image patch to be extracted from full image.
        idRange2: list [idx2_low, idx2_hi, idx2_stride], Specifying the second index of the image patch to be extracted from full image.
        idRange3: list [idx3_low, idx3_hi, idx3_stride], Specifying the third index of the image patch to be extracted from full image.

    '''

    idRange1_vect = range(idRange1[0], idRange1[1], idRange1[2])
    idRange2_vect = range(idRange2[0], idRange2[1], idRange2[2])
    idRange3_vect = range(idRange3[0], idRange3[1], idRange3[2])

    patch_3D = img_3D[np.ix_(idRange1_vect, idRange2_vect, idRange3_vect)] # z y x
    #print("getPatch_3D: patch_3D shape = ", np.shape(patch_3D))
    return patch_3D

def getPatchList_3D(recon3D, params):
    ''' convert 3D full sized image of size (Nx,Ny,Nz) to a list of patches with size params['patch_sizes']
    '''
    patchList3D = []
    print("getPatchList_3D: recon3D shape = ", np.shape(recon3D))
    for i_z in range(0, recon3D.shape[0] - params['stride'][1]*params['patch_sizes'][1] + 1, params['patch_strides'][1]):
        for i_y in range(0, recon3D.shape[1] - params['stride'][2]*params['patch_sizes'][2] + 1, params['patch_strides'][2]):
            for i_x in range(0, recon3D.shape[2] - params['stride'][3]*params['patch_sizes'][3] + 1, params['patch_strides'][3]):
                temp_patch = getPatch_3D(recon3D, [i_z, i_z+params['stride'][1]*params['patch_sizes'][1], params['stride'][1]], [i_y,i_y+params['stride'][2]*params['patch_sizes'][2], params['stride'][2]], [i_x,i_x+params['stride'][3]*params['patch_sizes'][3], params['stride'][3]])
                patchList3D.append(temp_patch)
    return patchList3D

def getPatchList_3D_rotAll(recon3D, params):

    if params['recon_augment']=='all_xy_3D':
        # all rotation augmentations (3 cases) assuming conv along x,y
        patchList3D = []

        # all possible positions for xy (dim=1,2)
        patchList3D = patchList3D + getPatchList_3D(np.transpose(recon3D, (0, 1, 2)), params)
        patchList3D = patchList3D + getPatchList_3D(np.transpose(recon3D, (1, 2, 0)), params)
        patchList3D = patchList3D + getPatchList_3D(np.transpose(recon3D, (2, 1, 0)), params)

    elif params['recon_augment']=='none':
        # all rotation augmentations (3 cases) assuming conv along x,y
        patchList3D = []

        # just xy
        patchList3D = patchList3D + getPatchList_3D(np.transpose(recon3D, (0, 1, 2)), params)
        
    else:
        error('getPatchList_3D_rotAll: undefined recon_augment string')

    return patchList3D

def get_patches_from_recon4D(reconList, params):

    # reconList = read3Dlist(fileList_recon4D, isRecon=True)

    stack_of_patchList3D = []
    for id,recon in enumerate(reconList):
        
        patchList3D = getPatchList_3D_rotAll(recon, params) # z y x
        stack_of_patchList3D.append(patchList3D)

        # print('Getting 3D patches from 3D volume number: {}'.format(id))
        # print('Volume size: {}'.format(recon.shape))
        # print('Patch size: {}'.format(patchList3D[0].shape))
        # print('Number of patches: {}'.format(len(patchList3D)))

    patchList_4D = []
    for patchId in range(len(stack_of_patchList3D[0])):

        patch4D_full_as_list = [patchList3D[patchId] for patchList3D in stack_of_patchList3D]
        patch4D_full = np.stack(patch4D_full_as_list, axis=0) # t z y x
        

        for i_t in range(0, len(patch4D_full_as_list) - params['patch_sizes'][0] + 1, params['patch_strides'][0]):

            patch4D = patch4D_full[i_t:i_t+params['patch_sizes'][0],:,:,:] # t z y x

            patch4D = np.transpose(patch4D, axes=params['patch_permute'])

            patch4D = np.squeeze( patch4D, tuple(params['patch_squeezeDim']) )

            patchList_4D.append(patch4D)

        if patchId==0:
            print('4D patch full shape: {}'.format(patch4D_full.shape))
            print('4D patch shape: {}'.format(patch4D.shape))


    print('#######################################################')
    print('Number of 3D patches: {}'.format(len(patchList_4D)))
    print('Shape of 3D patches: {}'.format(patchList_4D[0].shape))

    return patchList_4D


def get_patches_from_stacked_recon4D(load_path, params):

    dirPathList = readFileList(load_path)
    #list_of_fileList_recon4D = read_stackOfFileNames(dirPathList, 'recon')
    list_of_fileList_recon4D = read_stackOfFileNames(dirPathList, 'npy')

    patchList = []

    for fileList_recon4D, dirPath in zip(list_of_fileList_recon4D,dirPathList):

        print('Getting patches from stack of 4D volumes at: {}'.format(dirPath))

        patchList = patchList + get_patches_from_recon4D(fileList_recon4D, params) # conv1 x conv2 x chan

    return patchList


def main():
    print("entering main function of patch generation")
    np.random.seed(seed=1)

    params = get_params(args['params_path'])

    os.makedirs(os.path.dirname(params['paths']['patches_train_clean']), exist_ok=True)
    os.makedirs(os.path.dirname(params['paths']['patches_train_noisy']), exist_ok=True)

    patchList_clean = get_patches_from_stacked_recon4D(params['paths']['reconList_train_clean'], params)
    patchList_noisy = get_patches_from_stacked_recon4D(params['paths']['reconList_train_noisy'], params)
    print("number of clean patches in List = ", len(patchList_clean))
    assert len(patchList_clean)==len(patchList_noisy), 'error: number of clean and noisy patches unequal'
    assert patchList_clean[0].shape==patchList_noisy[0].shape, 'error: shape of clean and noisy patches unequal'


    patchList_clean_good = []
    patchList_noisy_good = []
    for patch_clean,patch_noisy in zip(patchList_clean, patchList_noisy):

        if isPatchGood(patch_clean, params):
            patchList_clean_good.append(patch_clean)
            patchList_noisy_good.append(patch_noisy)

    patchArray_clean = np.stack(patchList_clean_good, axis=0) # n x conv1 x conv2 x chan
    patchArray_noisy = np.stack(patchList_noisy_good, axis=0) # n x conv1 x conv2 x chan
    print("patchArray_clean shape: {}".format(patchArray_clean.shape))
    print("patchArray_noisy shape: {}".format(patchArray_noisy.shape))
    print("saving clean patch data to ", params['paths']['patches_train_clean'])
    np.save(params['paths']['patches_train_clean'], patchArray_clean)
    np.save(params['paths']['patches_train_noisy'], patchArray_noisy)

    save_summary(patchArray_clean, folderName=os.path.join(params['paths']['out_dir'], 'patch_summary_gen_clean'), num_patch_summary=params['num_patch_summary'], verbose=0)
    save_summary(patchArray_noisy, folderName=os.path.join(params['paths']['out_dir'], 'patch_summary_gen_noisy'), num_patch_summary=params['num_patch_summary'], verbose=0)

if __name__ == '__main__':
    main()

