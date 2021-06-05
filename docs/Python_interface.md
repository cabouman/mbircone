
Use swapaxes to make arrays correct shape

===========================================================================
Plan for implementation of the interface

- remove bash stuff
- rename c top level
- rename cython top level
- update setup.py
- Install dummy py interface

1) Implement utility function compute_sino_params

2) Implement utility function compute_img_params

3) Convert geometry parameters of demo to simple form to test utility functions

4) Populate recon function by calling geometry utilities and cython recon

5) Call regularization based utilities in recon function

===========================================================================


auto_roi_radius

Change automatically set to automatically set by [routine]
