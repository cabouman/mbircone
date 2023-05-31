========
Overview
========


**mbircone** is a Python implementation of the MBIR cone beam reconstruction (Model Based Iterative Reconstruction) algorithm :cite:`Balke2018`.
The code performs Bayesian reconstruction of tomographic data, so it is particularly well-suited for sparse view reconstruction from noisy data.
It also has hooks to support Plug-and-Play prior models that can dramatically improve image quality :cite:`venkatakrishnan2013plug` :cite:`sreehari2016plug`.
The reconstruction engine for *svmbir* is written and optimized in C, and it uses a thin Cython middleware layer between the Python interface and the C computation engine.
This allows for efficient memory transfers between Python and C.

**Geometry**

**mbircone** supports both *cone-beam* and *paralle-beam laminography* imaging geometries.
See the diagrams below for the different fan specifications.

.. list-table::

    * - .. figure:: figs/geom-cone-beam.png
           :align: center

           Cone-Beam geometry

      - .. figure:: figs/geom-laminography.png
           :align: center

           Parallel-Beam Laminography geometry

**Note on view angle ordering**

In certain imaging systems with slow acquisition, it is common practice to collect view data using techniques such as the "golden ratio" method in which the view angles are not collected in monotonically increasing order on the interval :math:`[0,2\pi)`. While ``svmbir`` will produce the correct reconstruction regardless of view ordering, its reconstruction speed will be substantially degraded when the views are not in monotone order. In this case, we highly recommend that users reorder the sinogram views using the provided ``sino_sort`` function. The ``sino_sort``  function first wraps the view angles modulo :math:`2\pi`, and then sorts the views to be in monotonically increasing order by view angle.


**Conversion from Arbitrary Length Units (ALU)**

In order to simplify usage, reconstructions are done using arbitrary length units (ALU). In this system, 1 ALU can correspond to any convenient measure of distance chosen by the user. So for example, it is often convenient to take 1 ALU to be the distance between pixels, which by default is also taken to be the distance between detector channels.


*Transmission CT Example:* For this example, assume that the physical spacing between detector channels is 5 mm. In order to simplify our calculations, we also use the default detector channel spacing and voxel spacing of ``delta_channel=1.0`` and ``delta_xy=1.0``. In other words, we have adopted the convention that the voxel spacing is 1 ALU = 5 mm, where 1 ALU is now our newly adopted measure of distance.

Using this convention, the 3D reconstruction array, ``image``, will be in units of :math:`\mbox{ALU}^{-1}`. However, the image can be converted back to more conventional units of :math:`\mbox{mm}^{-1}` using the following equation:

.. math::

    \mbox{image in mm$^{-1}$} = \frac{ \mbox{image in ALU$^{-1}$} }{ 5 \mbox{mm} / \mbox{ALU}}


*Emission CT Example:* Once again, we assume that the channel spacing in the detector is 5 mm, and we again adopt the default reconstruction parameters of ``delta_channel=1.0`` and ``delta_xy=1.0``. So we have that 1 ALU = 5 mm.

Using this convention, the 3D array, ``image``, will be in units of photons/AU. However, the image can be again converted to units of photons/mm using the following equation:

.. math::

    \mbox{image in photons/mm} = \frac{ \mbox{image in photons/ALU} }{ 5 \mbox{mm} / \mbox{ALU}}

**Matrix caching**

When system matrices are computed, they are stored to disk and will be automatically loaded whenever the same geometry is subsequently encountered.
By default, the system matrices are stored in the subfolder ``~/.cache/mbircone/sysmatrix`` of your home directory.
The matrix files can be removed at any time, and should be periodically cleaned out to reduce disk use.
Occasionally, updates to the software package include changes to the encoding of the system matrix, in which case the the cached matrix files should also be cleaned out to avoid incompatibility.

