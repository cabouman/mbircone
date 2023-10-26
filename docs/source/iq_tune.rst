====================
Image Quality Tuning
====================

An important feature of **mbircone** is that it automatically sets reconstruction paramers using practical heuristics that should get you close to the best image quality (IQ).
However, some minor tuning may be required to get the best image quality for your application.


User IQ Tuning
--------------

While the automatic parameter setting should be close, many user may want to further tune IQ to suite their needs.
Here is our recommended strategy for tuning image quality.

**Step 1: Adjust Image Sharpness:**
In order to adjust the sharpness of the image, we recommend the you first adjust the value of the ``sharpness`` parameter.
The default value of ``sharpness`` is 0.0. Larger positive values will increase image sharpness, and negative values will decrease the image sharpness.
For most users, this is all that will be needed to achieve the best image quality.

Alternatively, you can also change the sharpness by changing the value of ``snr_db``, the variable that controls the assumed signal-to-noise ratio of the data.
Increasing ``snr_db`` above its default value of 40.0 will increase image sharpness and reducing it will decrease sharpness.

Note that both the ``sharpness`` and ``snr_db`` parameters work for both the standard reconstruction mode with the qGGMRF prior and the proximal map mode.

**Step 2: Adjust Image Edginess:**
Advanced users may want to increase the sharpness of edges without changing the overall sharpness of the image.
This can be done by adjusting the value of the parameter ``T``.
The default value of ``T`` is 1.0. Smaller positive values will result in an image that has sharper low contrast edges.
Larger values of ``T`` will ultimately result uniform quadratic regularization across the image.

**Step 3: Adjust Image Edginess:**
Very advanced users may also want to change the value of the parameter ``p``.
The default value of ``p`` is 1.2, and it should alway be choosen to be >1.0.
As ``p`` becomes smaller, the reconstruction will become closer to a total-variation regularized reconstruction.
However, many users report that such reconstructions tend to look waxy.




Automatic Parameter Selection
-----------------------------

One of the major barriers to use of iterative reconstruction is the selection of regularization parameters.
While the selection of these parameters is crucial to achieving good IQ, most users find these parameters extremely difficult to set.
Below we give a summary of how the major MBIR parameters are automatically set based on the amount, quality, dimension of the CT data,
as well as the relative resolution of the reconstruction.


**``sigma_y``:** This parameter controls the value of the noise standard deviation variable, :math:`\sigma_y`, described in the
[Theory Section](https://mbircone.readthedocs.io/en/latest/theory.html#forward-model).
The value of this parameter is set by the function ``auto_sigma_y``, using the following mathematical formula

.. math::

    \sigma_y = 10^{-\mbox{snr_db}/20} * (\mbox{RMS Sinogram Amplitude}) * R^{1/2}

where ``snr_db`` has a default value of 40.0 determines the assuming signal-to-noise ratio of the CT sinogram data,
and :math:`R` is the ratio of the reconstructed voxel pitch to the default pitch.
The final term accounts compensates for the increased regularization that occurs as the resolution of the reconstruction is decreased.


**``sigma_x``:** This parameter controls the value of the image scale variable, :math:`\sigma_x`, described in the
[Theory Section](https://mbircone.readthedocs.io/en/latest/theory.html#prior-model).
The value of this parameter is set by the function ``auto_sigma_x``, using the following mathematical formula

.. math::

    \sigma_x = 0.2 * (\mbox{Average Sinogram Amplitude}) * / \left[ (\mbox{Detector Width}) / M \right]

where :math:`M` is the magnification of the cone-beam system.
The constant of 0.2 accounts for the fact that the difference between neighboring pixels should be relatively small compared to the average value of the reconstruction.



