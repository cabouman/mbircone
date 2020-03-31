
%% Parameter setting %%

% % % % % % Confirm your parameters % % % % % % %

param.nx = imgParams.N_x; % number of voxels
param.ny = imgParams.N_y;
param.nz = imgParams.N_z;

%The real detector panel pixel density (number of pixels)
param.nu = sinoParams.N_dv;     % number of pixels
param.nv = sinoParams.N_dw;

param.dx = imgParams.Delta_xy; % single voxel size
param.dy = imgParams.Delta_xy;
param.dz = imgParams.Delta_z;
param.du = sinoParams.Delta_dv;
param.dv = sinoParams.Delta_dw;

% Image setting (real size)
param.sx = param.nx * param.dx; % mm (real size)
param.sy = param.ny * param.dy; % mm
param.sz = param.nz * param.dz; % mm

% Detector setting (real size)
param.su = param.nu * param.du;    % mm (real size)
param.sv = param.nv * param.dv;     % mm

% X-ray source and detector setting
param.DSD = sinoParams.u_d0 - sinoParams.u_s;    %  Distance source to detector 
param.DSO = -sinoParams.u_s;   %  X-ray source to object axis distance

% angle setting
param.dir = -1;   % gantry rotating direction (clock wise/ counter clockwise)
param.deg = param.dir * angleList/(2*pi)*360; % you can change
param.nProj = length(param.deg);
param.dang = mean(diff(sort(param.deg)));
param.parker = 1; % data with 360 deg -> param.parker = 0 , data less than 360 deg -> param.parker=1 



% % % % % % Confirm your parameters % % % % % % %
 
% filter='ram-lak','cosine', 'hamming', 'hann' 
param.filter='ram-lak'; % high pass filter

% % % Geometry calculation % % %
param.xs = imgParams.x_0 + ((0:param.nx-1)+0.5)*param.dx;
param.ys = imgParams.y_0 + ((0:param.ny-1)+0.5)*param.dy;
param.zs = imgParams.z_0 + ((0:param.nz-1)+0.5)*param.dz;

param.us = sinoParams.v_d0 + ((0:param.nu-1)+0.5)*param.du;
param.vs = sinoParams.w_d0 + ((0:param.nv-1)+0.5)*param.dv;

param.interptype = 'linear'; % 'linear', 'nearest'

% % % % % % Confirm your parameters % % % % % % %
% Only for Matlab version above 2013b with parallel computing toolbox: Speed dependent on your GPU
% You don't need to install CUDA, but install latest graphics driver.
% only Nvidia GPU cards can use this. otherwise please "param.gpu=0"
% This option is semi-GPU code using built-in Matlab GPU functions: several times faster than CPU
param.gpu = 0;


















