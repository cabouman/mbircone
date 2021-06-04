
Use swapaxes to make arrays correct shape

===========================================================================

center offset: offline discussion


center_offset:
Distance to the center of the detector from the projection of the X-ray source at the detector
Distance to the center of the detector point in the detector closest to the X-ray source.

rotation_offset:
Shortest distance between the rotation axis and the line though the X-ray source that is perperndicular to the detector in units of ALU.


===========================================================================
Plan for implementation of the interface

1) Implement utility function compute_sino_params

2) Implement utility function compute_img_params

3) Convert geometry parameters of demo to simple form to test utility functions

4) Populate recon function by calling geometry utilities and cython recon

5) Call regularization based utilities in recon function

===========================================================================


change horizontal/vertical to channel/rows

source-to-detector line define in md file

auto_roi_radius

Change automatically set to automatically set by [routine]
