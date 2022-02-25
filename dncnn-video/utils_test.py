
import os
import sys
import argparse
import random
import numpy as np
from PIL import Image
import array
import pdb

from utils import *

#############################################################################
## Load/Save Data
#############################################################################

class loader_reconList:
    def __init__(self, img_list, size_z_in=1, size_z_out=1):

        self.size_z_in = size_z_in
        self.size_z_out = size_z_out

        assert len(img_list) != 0, 'img_list empty'

        self.inData = []
        self.outData = []
        self.padSize_in1, self.padSize_in2 = semi2DCNN_inPad(self.size_z_in, self.size_z_out)

        for i,recon in enumerate(img_list):

            assert recon.shape[0]%self.size_z_out == 0, 'shape_z({}) not divisible by size_z_out({})'.format(recon.shape[0],self.size_z_out)

            recon_out = np.copy(recon)
            self.outData.append(recon_out)

            recon_in = np.pad(np.copy(recon), ((self.padSize_in1,self.padSize_in2), (0,0), (0,0)), 'reflect')
            self.inData.append(recon_in)

    def __iter__(self):

        for self.id_t in range(len(self.outData)):

            # iterate in & out data with stride size_z_out starting at 0
            z_out_iter = range(0,self.outData[self.id_t].shape[0],self.size_z_out)

            for (self.id_z_in,self.id_z_out) in zip(z_out_iter,z_out_iter):
                
                yield self.getInput_current()

    def getInput_current(self):

        return self.getInput(self.id_t, self.id_z_in)

    def setOutput_current(self, img):

        return self.setOutput(self.id_t, self.id_z_out, img)

    def getInput(self, id_t, id_z_in):

        img = self.inData[id_t][range(id_z_in,id_z_in+self.size_z_in)] # z x n1 x n2
        img = np.swapaxes(img, 0, 2) # n2 x n1 x z
        img = np.swapaxes(img, 0, 1) # n1 x n2 x z
        img = np.expand_dims(img, axis=0) # 1 x n1 x n2 x z
        img = np.copy(img, order='C') # contiguous order

        return img

    def setOutput(self, id_t, id_z_out, img):
        # shape: 1 x n1 x n2 x z

        img = np.squeeze(img, axis=0) # n1 x n2 x z
        img = np.swapaxes(img, 0, 1) # n2 x n1 x z
        img = np.swapaxes(img, 0, 2) # z x n1 x n2
        self.outData[id_t][range(id_z_out,id_z_out+self.size_z_out)] = img


def compute_inv_permute_vector(permute_vector):

    inv_permute_vector = []
    for i in range(len(permute_vector)):
        # print('i = {}'.format(i))
        position_of_i = permute_vector.index(i)
        # print('position_of_i = {}'.format(position_of_i))
        inv_permute_vector.append(position_of_i)

    return inv_permute_vector

