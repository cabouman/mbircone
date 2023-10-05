========
Overview
========

**mbircone** is a Python/C implementation of the MBIR (Model Based Iterative Reconstruction) cone beam reconstruction algorithm :cite:`balke2018separable`.
The code performs Bayesian reconstruction of tomographic data, so it is particularly well-suited for sparse view reconstruction from noisy data.
It also has hooks to support Plug-and-Play prior models that can dramatically improve image quality :cite:`venkatakrishnan2013plug` :cite:`sreehari2016plug`.
The reconstruction engine for *mbircone* is written and optimized in C. It uses a thin Cython middleware layer between the Python interface and the C computation engine, which allows for efficient memory transfers between Python and C.


Geometry
--------

**mbircone** supports both *cone-beam* and *paralle-beam laminography* imaging geometries, illustrated in the diagram below.

.. list-table::

    * - .. figure:: figs/geom-cone-beam.png
           :align: center

           Cone-Beam geometry

      - .. figure:: figs/geom-laminography.png
           :align: center

           Parallel-Beam Laminography geometry


Arbitrary Length Units (ALU)
----------------------------

In order to simplify usage, reconstructions are done using arbitrary length units (ALU).
In this system, 1 ALU can correspond to any convenient measure of distance chosen by the user.
So for example, it is often convenient to take 1 ALU to be the distance between pixels, which by default is also taken to be the distance between detector channels.

The following two examples show how to convert from ALU to physical units for transmission and emission cases.

*Transmission CT Example:* For this example, assume that the physical spacing between detector channels is 5 mm. In order to simplify our calculations, we also use the default detector channel spacing and voxel spacing of ``delta_channel=1.0`` and ``delta_xy=1.0``. In other words, we have adopted the convention that the voxel spacing is 1 ALU = 5 mm, where 1 ALU is now our newly adopted measure of distance.

Using this convention, the 3D reconstruction array, ``image``, will be in units of :math:`\mbox{ALU}^{-1}`. However, the image can be converted back to more conventional units of :math:`\mbox{mm}^{-1}` using the following equation:

.. math::

    \text{image in mm$^{-1}$} = \frac{ \text{image in ALU$^{-1}$} }{ 5 \text{mm} / \text{ALU}}


*Emission CT Example:* Once again, we assume that the channel spacing in the detector is 5 mm, and we again adopt the default reconstruction parameters of ``delta_channel=1.0`` and ``delta_xy=1.0``. So we have that 1 ALU = 5 mm.

Using this convention, the 3D array, ``image``, will be in units of photons/AU. However, the image can be again converted to units of photons/mm using the following equation:

.. math::

    \text{image in photons/mm} = \frac{ \text{image in photons/ALU} }{ 5 \text{mm} / \text{ALU}}


Matrix caching
--------------

When system matrices are computed, they are stored to disk and will be automatically loaded whenever the same geometry is subsequently encountered.
By default, the system matrices are stored in the subfolder ``~/.cache/mbircone`` of your home directory.
The matrix files can be removed at any time, and should be periodically cleaned out to reduce disk use.
Occasionally, updates to the software package include changes to the encoding of the system matrix, in which case the the cached matrix files should also be cleaned out to avoid incompatibility.
