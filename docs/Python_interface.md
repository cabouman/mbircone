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

Uni-directional arrow in diagram
center offset: offline discussion
set u_r to zero in fig

is_qggmrf remove

num_neighbors default to 6
Specify 6 is fastest, 

sigma_proxmap remove

plan for implementation of the interface