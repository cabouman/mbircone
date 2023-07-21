============
Installation
============

The ``mbircone`` package is currently only available to download and install from source available from `GitHub <https://github.com/cabouman/mbircone>`_.


Downloading and installing from source
--------------------------------------

1. Download the source code:

  In order to download the C and python code, move to a directory of your choice and run the following two commands.

	| ``git clone https://github.com/cabouman/mbircone.git``
	| ``cd mbircone``


2. Create a Virtual Environment:

  It is recommended that you install to a virtual environment.
  If you have Anaconda installed, you can run the following:

	| ``conda create --name mbircone python=3.8``
	| ``conda activate mbircone``

  Install the dependencies using:

	``pip install -r requirements.txt``

  Before using the package, this ``mbircone`` environment needs to be activated.


3. Install:

The ``mbircone`` package requires a C compiler together with OpenMP libraries for parallel multicore processing.
The four supported compilers are the open source ``gcc`` compiler, Microsoft Visual C ``msvc``, Intel's ``icc`` compiler, or the Apple's ``clang`` compiler.
The Intel compiler currently offers the best performance on x86 processors through the support of the AVX instruction set;
however, the ``gcc`` and ``clang`` compilers are often more readily available.

**Important:** You must first install one of these compilers together with the associated OpenMP libraries on your computer.
MacOS and Windows users should refer to the instructions :ref:`below <Windows and Mac>` for more details on installation of the compilers, OMP libraries and associated utilities.

Once the compiler and OMP libraries are installed, the following commands can be used to compile the ``mbircone`` code.

For installation using the four possible compiler options, run one of the following in a bash shell:

``CC=gcc pip install .``

``CC=icc pip install .``

``CC=clang pip install .``

``CC=msvc pip install .``

In each case, the commands should be run from the root directory of the repository.
Also, see the sections below for trouble shooting tips for installing under the different operating systems.

You can verify the installation by running ``pip show mbircone``, which should display a brief summary of the packages installed in the ``mbircone`` environment.
Now you will be able to use the ``mbircone`` python commands from any directory by running the python command ``import mbircone``.


.. _Windows and Mac:

Installation on Windows and MacOS
---------------------------------

Below are some tips for compiling and running the package under the Windows and MacOSx operating systems.
Linux is more straight forward.

1. *Intel icc Compiler:*
The Intel compiler and OMP libraries when coupled with the appropriate Intel x86 processor
can substantially increase ``mbircone`` performance by enabling the AVX2 instructor set.
The ``icc/OpenMP`` compiler and libraries exists for Linux, Windows, and MacOS, but may need to be purchased.
The icc compiler is available `[here] <https://software.intel.com/content/www/us/en/develop/tools/parallel-studio-xe.html>`__.

2. *Windows Installation:* The package will run under Windows, but there tend to be more things that can go wrong due to the wide variety of possible configurations. The following list of recommended configurations have been tested to work, but others are possible:

* *64-bit gcc or Intel icc compiler:* For the command line version, make sure to install a 64bit compiler such as the ``MinGW_64`` available from `[here] <http://winlibs.com>`__ or the Intel ``icc`` compiler as described above. Commonly used gcc compilers are only 32bit and will create ``calloc`` errors when addressing array sizes greater than 2Gb.

* *MinGW + MSYS environment:* For the command line version, we recommend installing ``MinGW`` including the ``msys`` utilities. These utilities support a minimalist set of traditional UNIX tools.

* *Git Bash:* We recommend installing `[Git Bash] <https://gitforwindows.org>`__ to support bash scripting.

One known issue is that in some Windows bash environments the C executable ``mbir_ct.exe`` may not be properly moved to the ``bin`` directory.
If this occurs, then the problem can be resolved by manually moving the file.

3. *MacOS Installation:*
MacOS users will typically use the ``clang`` compiler provided as part of the Xcode Developer Tools.
In this case, the ``gcc`` command in the MacOS environment is **not** actually ``gcc``.
Instead it is an alias to the ``clang`` compiler.
Therefore, the C code should be compiled using the ``clang`` option.

In order to obtain ``clang`` you will need to install the most up-to-date version of both Xcode
and ``Command Line Tools for Xcode`` available `[here] <https://developer.apple.com/download/more/>`__.

Importantly, the Xcode Developer tools **do not include** the required OpenMP libraries.
The OMP libraries can be obtained from `[here] <https://mac.r-project.org/openmp/>`__.
You will need to download a file of the form ``openmp-XXX.tar.gz``.
The tar file will contain the following files::

    /usr/local/lib/libomp.dylib
    /usr/local/include/ompt.h
    /usr/local/include/omp.h
    /usr/local/include/omp-tools.h


These files should be moved to the specified directories.
You may also need to open the file ``/usr/local/lib/libomp.dylib``.
This will generate a splash screen that requests permision of OSx to execute the library.

In addition, after OS updates, you may need to reinstall the Xcode toolkit using the command: ``xcode-select --install``
