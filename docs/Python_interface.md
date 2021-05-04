System matrix: disk / compute / compute and store as variable

Use swapaxes to make arrays correct shape

Set optional params

How to parameterize geometry params in a unitless fashion
(magnification, dist(beam_center,det_denter))

dist_source_detector,
magnification

Optional:
center_offset (distance between detector center and center of beam) default [0, 0] ALU
rotation_offset (distance between object center and axis of rotatation) default [0] ALU
delta_pixel_detector, (default: 1ALU)
delta_pixel_image, (default: 1ALU/magnification)

Divide params into groups (required, optional, internal)

===========================================================================

Required:

```
sino, angles, 
dist_source_detector, magnification
```


Optional:
```
center_offset=(0,0), rotation_offset=0, delta_pixel_detector=1, delta_pixel_image=1,
init_image=0.0, prox_image=None,
sigma_y=None, snr_db=30.0, weights=None, weight_type = 'unweighted',
is_qggmrf=True, is_proxmap=False, is_positivity_constraint=True, 
q=2, p=1.2, T=2, num_neighbors=26,
sigma_x=None, sigma_proxmap=None, max_iterations=20,
num_threads=None, 
is_NHICD=False,
verbose=False,
lib_path='~/.cache/mbircone'
```

Internal:

```
NHICD_ThresholdAllVoxels_ErrorPercent=80, NHICD_percentage=15, NHICD_random=20, 
zipLineMode=2, N_G=2, numVoxelsPerZiplineMax=200
```