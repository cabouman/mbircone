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
PACKAGES = [PACKAGE_DIR]

# Single threaded gcc compile; tested for MacOS and Linux

# Single threaded clang compile; tested for MacOS and Linux
if os.environ.get('CC') == 'clang':
    c_extension = Extension(PACKAGE_DIR+".conebeam",
                      # [PACKAGE_DIR+"/src/matrices.c", 
                      [PACKAGE_DIR+"/src/allocate.c", PACKAGE_DIR + "/conebeam.pyx"],
                      libraries=[],
                      include_dirs=[np.get_include()])

# OpenMP gcc compile: tested for MacOS and Linux
if os.environ.get('CC') =='gcc':
    c_extension = Extension(PACKAGE_DIR+".conebeam",
                      # [PACKAGE_DIR+"/src/matrices.c", 
                      [PACKAGE_DIR+"/src/allocate.c", PACKAGE_DIR + "/conebeam.pyx"],
                      libraries=[],
                      include_dirs=[np.get_include()],
                      # for gcc-10 "-std=c11" can be added as a flag
                      extra_compile_args=["-O3", "-fopenmp","-Wno-unknown-pragmas"],
                      extra_link_args=["-lm","-fopenmp"]) 


# OpenMP icc compile: tested for MacOS and Linux
if os.environ.get('CC') =='icc':
    c_extension = Extension(PACKAGE_DIR+".conebeam",
                      # [PACKAGE_DIR+"/src/matrices.c", 
                      [PACKAGE_DIR+"/src/allocate.c", PACKAGE_DIR + "/conebeam.pyx"],
                      libraries=[],
                      include_dirs=[np.get_include()],
                      # for gcc-10 "-std=c11" can be added as a flag
                      extra_compile_args=["-DICC","-qopenmp","-no-prec-div", "-restrict" ,"-ipo","-inline-calloc",
                                          "-qopt-calloc","-no-ansi-alias","-xCORE-AVX2"],
                      extra_link_args=["-lm","-DICC","-qopenmp","-no-prec-div", "-restrict" ,"-ipo","-inline-calloc",
                                          "-qopt-calloc","-no-ansi-alias","-xCORE-AVX2"])


setup(install_requires=REQUIRES,
      packages=PACKAGES,
      zip_safe=False,
      name=NAME,
      version=VERSION,
      description=DESCRIPTION,
      author=AUTHOR,
      license=LICENSE,
      cmdclass={"build_ext": build_ext},
      ext_modules=[c_extension]
      )