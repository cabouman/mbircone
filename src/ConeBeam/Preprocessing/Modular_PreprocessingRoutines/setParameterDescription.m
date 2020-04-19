function [ name ] = setParameterDescription( )

name.N_dv = 'number of detector pixels (N_dv) in v-direction (horizontal) in units of [1]';
name.N_dw = 'number of detector pixels (N_dw) in w-direction (vertical) in units of [1]';
name.N_beta = 'number of views (N_beta) in units of [1]';
name.Delta_dv = 'detector spacing (Delta_dv) in v-direction (horizontal) in units of [mm]';
name.Delta_dw = 'detector spacing (Delta_dw) in w-direction (vertical) in units of [mm]';
name.u_s = 'x-ray source location (u_s) on u-axis (perpendicular to detector) in units of [mm]';
name.u_r = 'u-coordinate of rotation axis (u_r) in units of [mm]';
name.v_r = 'v-coordinate of rotation axis (v_r) in units of [mm]';
name.u_d0 = 'u-coordinate of all detector pixels (u_d0) in units of [mm]';
name.v_d0 = 'v-coordinate of first detector pixel (v_d0) in units of [mm]';
name.w_d0 = 'w-coordinate of first detector pixel (w_d0) in units of [mm]';

name.x_0 = 'x-coordinate of the first voxel (x_0) in units of [mm]';
name.y_0 = 'y-coordinate of the first voxel (y_0) in units of [mm]';
name.z_0 = 'z-coordinate of the first voxel (z_0) in units of [mm]';
name.N_x = 'number of voxels in x-dimension (N_x) in units of [1]';
name.N_y = 'number of voxels in y-dimension (N_y) in units of [1]';
name.N_z = 'number of voxels in z-dimension (N_z) in units of [1]';
name.Delta_xy = '(Delta_xy) side length of a voxel in xy plane in units of [mm]';
name.Delta_z  = '(Delta_z)  side length of a voxel in z direction in units of [mm]';

name.x_0_roi = '(x_0_roi) coordinate of the first voxel of the roi in the x-direction (units of [mm])';
name.y_0_roi = '(y_0_roi) coordinate of the first voxel of the roi in the y-direction (units of [mm])';
name.z_0_roi = '(z_0_roi) coordinate of the first voxel of the roi in the z-direction (units of [mm])';

name.j_xstart_roi = '(j_xstart_roi) index of the first voxel of the roi in the x direction';
name.j_ystart_roi = '(j_ystart_roi) index of the first voxel of the roi in the y direction';
name.j_zstart_roi = '(j_zstart_roi) index of the first voxel of the roi in the z direction';

name.j_xstop_roi = '(j_xstop_roi) index of the last voxel of the roi in the x direction';
name.j_ystop_roi = '(j_ystop_roi) index of the last voxel of the roi in the y direction';
name.j_zstop_roi = '(j_zstop_roi) index of the last voxel of the roi in the z direction';



end

