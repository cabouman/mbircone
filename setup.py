import os
import sys
import numpy as np
from setuptools import setup, Extension
from Cython.Distutils import build_ext

NAME = "mbircone"
VERSION = "0.1b1"
DESCRIPTION = "Python Package for Cone Beam reconstruction"
REQUIRES = ['numpy','Cython','psutil','Pillow']  # external package dependencies
LICENSE = "BSD-3-Clause"
AUTHOR = "Soumendu Majee"


# Specifies directory containing cython functions to be compiled
PACKAGE_DIR = "mbircone"

SRC_FILES = [PACKAGE_DIR + '/src/allocate.c', PACKAGE_DIR + '/src/MBIRModularUtilities3D.c',
             PACKAGE_DIR + '/src/icd3d.c', PACKAGE_DIR + '/src/recon3DCone.c',
             PACKAGE_DIR + '/src/icd3dDenoise.c', PACKAGE_DIR + '/src/denoise3D.c',
             PACKAGE_DIR + '/src/computeSysMatrix.c',
             PACKAGE_DIR + '/src/interface.c', PACKAGE_DIR + '/interface_cy_c.pyx']


compiler_str = os.environ.get('CC')

# Set default to gcc in case CC is not set
if not compiler_str:
  compiler_str = 'gcc'
  os.environ['CC']='gcc'
  print("\n*** CC is not set. Using GCC as default compiler. ***\n")

# Single threaded clang compile
if compiler_str == 'clang':
    print("\n*** Using CLANG as compiler. ***\n")
    c_extension = Extension(PACKAGE_DIR+'.interface_cy_c', SRC_FILES,
                    libraries=[],
                    language='c',
                    include_dirs=[np.get_include()])


# OpenMP gcc compile
if compiler_str =='gcc':
    print("\n*** Using GCC as compiler. ***\n")
    c_extension = Extension(PACKAGE_DIR+'.interface_cy_c', SRC_FILES,
                    libraries=[],
                    language='c',
                    include_dirs=[np.get_include()],
                    # for gcc-10 "-std=c11" can be added as a flag
                    extra_compile_args=["-std=c11","-O3", "-fopenmp","-Wno-unknown-pragmas"],
                    extra_link_args=["-lm","-fopenmp"]) 


# OpenMP icc compile
if compiler_str =='icc':
    print("\n*** Using ICC as compiler. ***\n")
    if sys.platform == 'linux':
            os.environ['LDSHARED'] = 'icc -shared'
    c_extension = Extension(PACKAGE_DIR+'.interface_cy_c', SRC_FILES,
                    libraries=[],
                    language='c',
                    include_dirs=[np.get_include()],
                    extra_compile_args=["-O3","-DICC","-qopenmp","-no-prec-div","-restrict","-ipo","-inline-calloc",
                            "-qopt-calloc","-no-ansi-alias","-xCORE-AVX2"],
                    extra_link_args=["-lm","-qopenmp"])


setup(install_requires=REQUIRES,
      packages=[PACKAGE_DIR],
      zip_safe=False,
      name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      author=AUTHOR,
      license=LICENSE,
      cmdclass={"build_ext": build_ext},
      ext_modules=[c_extension]
      )
