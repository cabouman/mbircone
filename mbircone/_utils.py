# -*- coding: utf-8 -*-
# Copyright (C) 2020-2022 by MBIRCONE Developers
# All rights reserved. BSD 3-clause License.

import numpy as np
import os
import hashlib
import random
from PIL import Image

def hash_params(angles, sinoparams, imgparams):
    hash_input = str(sinoparams) + str(imgparams) + str(np.around(angles, decimals=6))
    hash_val = hashlib.sha512(hash_input.encode()).hexdigest()
    return hash_val

def _gen_sysmatrix_fname(lib_path, sysmatrix_name='object'):
    os.makedirs(os.path.join(lib_path, 'sysmatrix'), exist_ok=True)

    sysmatrix_fname = os.path.join(lib_path, 'sysmatrix', sysmatrix_name + '.sysmatrix')

    return sysmatrix_fname


def _gen_sysmatrix_fname_tmp(lib_path, sysmatrix_name='object'):
    sysmatrix_fname_tmp = os.path.join(lib_path, 'sysmatrix',
                                       sysmatrix_name + '_pid' + str(os.getpid()) + '_rndnum' + str(
                                           random.randint(0, 1000)) + '.sysmatrix')

    return sysmatrix_fname_tmp

