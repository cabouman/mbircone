from setuptools import setup, Extension
from Cython.Distutils import build_ext
import numpy as np
import os

NAME = "mbircone"
VERSION = "0.1"
DESCRIPTION = "Python Package for Cone Beam reconstruction"
REQUIRES = ['numpy', 'cython']
LICENSE = "BSD-3-Clause"
AUTHOR = "Soumendu Majee"


# Specifies directory containing cython functions to be compiled
PACKAGE_DIR = "mbircone"
SRC_FILES = [PACKAGE_DIR+'/src/allocate.c', PACKAGE_DIR+'/src/computeSysMatrix.c',
             PACKAGE_DIR+'/src/cyInterface.c', PACKAGE_DIR+'/src/icd3d.c',
             PACKAGE_DIR+'/src/MBIRModularUtilities3D.c', PACKAGE_DIR+'/src/recon3DCone.c',
             PACKAGE_DIR+'/conebeam.pyx']


compiler_str = os.environ.get('CC')

# Set default to gcc in case CC is not set
if not compiler_str:
  compiler_str = 'gcc'


# Single threaded clang compile
if compiler_str == 'clang':
    c_extension = Extension(PACKAGE_DIR+'.conebeam', SRC_FILES,
                      libraries=[],
                      language='c',
                      include_dirs=[np.get_include()])


# OpenMP gcc compile
if compiler_str =='gcc':
    c_extension = Extension(PACKAGE_DIR+'.conebeam', SRC_FILES,
                      libraries=[],
                      language='c',
                      include_dirs=[np.get_include()],
                      # for gcc-10 "-std=c11" can be added as a flag
                      extra_compile_args=["-O3", "-fopenmp","-Wno-unknown-pragmas"],
                      extra_link_args=["-lm","-fopenmp"]) 


# OpenMP icc compile
if compiler_str =='icc':
    c_extension = Extension(PACKAGE_DIR+'.conebeam', SRC_FILES,
                      libraries=[],
                      language='c',
                      include_dirs=[np.get_include()],
                      # for gcc-10 "-std=c11" can be added as a flag
                      extra_compile_args=["-DICC","-qopenmp","-no-prec-div", "-restrict" ,"-ipo","-inline-calloc",
                                          "-qopt-calloc","-no-ansi-alias","-xCORE-AVX2"],
                      extra_link_args=["-lm","-DICC","-qopenmp","-no-prec-div", "-restrict" ,"-ipo","-inline-calloc",
                                          "-qopt-calloc","-no-ansi-alias","-xCORE-AVX2"])


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