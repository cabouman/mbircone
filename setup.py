import os
import sys
import numpy as np
from setuptools import setup, Extension
from Cython.Distutils import build_ext

NAME = "mbircone"
VERSION = "0.1"
DESCRIPTION = "Python Package for Cone Beam reconstruction"
REQUIRES = ['numpy','Cython','psutil','Pillow']  # external package dependencies
LICENSE = "BSD-3-Clause"
AUTHOR = "Soumendu Majee"


# Specifies directory containing cython functions to be compiled
PACKAGE_DIR = "mbircone"
CONE3D_NAME = "cone3D"
PREPROCESS_NAME = "preprocess"
MACE_NAME = "mace"
CONE3D_DIR = PACKAGE_DIR + "/" + CONE3D_NAME
PREPROCESS_DIR = PACKAGE_DIR + "/" + PREPROCESS_NAME
MACE_DIR = PACKAGE_DIR + "/" + MACE_NAME
PYX_DIR = PACKAGE_DIR + "." + CONE3D_NAME

SRC_FILES = [CONE3D_DIR + '/src/allocate.c', CONE3D_DIR + '/src/MBIRModularUtilities3D.c',
             CONE3D_DIR + '/src/icd3d.c', CONE3D_DIR + '/src/recon3DCone.c',
             CONE3D_DIR + '/src/computeSysMatrix.c',
             CONE3D_DIR + '/src/interface.c', CONE3D_DIR + '/interface_cy_c.pyx']


compiler_str = os.environ.get('CC')

# Set default to gcc in case CC is not set
if not compiler_str:
  compiler_str = 'gcc'


# Single threaded clang compile
if compiler_str == 'clang':
    c_extension = Extension(PYX_DIR+'.interface_cy_c', SRC_FILES,
                    libraries=[],
                    language='c',
                    include_dirs=[np.get_include()])


# OpenMP gcc compile
if compiler_str =='gcc':
    c_extension = Extension(PYX_DIR+'.interface_cy_c', SRC_FILES,
                    libraries=[],
                    language='c',
                    include_dirs=[np.get_include()],
                    # for gcc-10 "-std=c11" can be added as a flag
                    extra_compile_args=["-std=c11","-O3", "-fopenmp","-Wno-unknown-pragmas"],
                    extra_link_args=["-lm","-fopenmp"]) 


# OpenMP icc compile
if compiler_str =='icc':
    if sys.platform == 'linux':
            os.environ['LDSHARED'] = 'icc -shared'
    c_extension = Extension(PYX_DIR+'.interface_cy_c', SRC_FILES,
                    libraries=[],
                    language='c',
                    include_dirs=[np.get_include()],
                    extra_compile_args=["-O3","-DICC","-qopenmp","-no-prec-div","-restrict","-ipo","-inline-calloc",
                            "-qopt-calloc","-no-ansi-alias","-xCORE-AVX2"],
                    extra_link_args=["-lm","-qopenmp"])


setup(install_requires=REQUIRES,
      packages=[PACKAGE_DIR,CONE3D_DIR,PREPROCESS_DIR,MACE_DIR],
      zip_safe=False,
      name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      author=AUTHOR,
      license=LICENSE,
      cmdclass={"build_ext": build_ext},
      ext_modules=[c_extension]
      )
