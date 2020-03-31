function [ parImg ] = computeImgParams( par, preprocessingParams )


%% Voxel size
M = (par.u_d0 - par.u_s) / ( -par.u_s);
parImg.Delta_xy = par.Delta_dv / M;
parImg.Delta_z  = par.Delta_dw / M;

%% Adjust Image parameters
parImg.Delta_xy = parImg.Delta_xy * preprocessingParams.scaler_Delta_xy;
parImg.Delta_z  = parImg.Delta_z  * preprocessingParams.scaler_Delta_z ;

%% Computation of x_0, y_0, N_x, N_y

v_d1 = par.v_d0 + par.N_dv * par.Delta_dv;

% %%%% Part 1: find radius of circle %%%%
% lower cone point P0
P0 = [par.u_d0; par.v_d0];

% upper cone point P1
P1 = [par.u_d0; v_d1];

% source point S
S = [par.u_s; 0];

% Rotation center point C
C = [par.u_r; par.v_r];

% r_0 = distance{ line(P0,S), C }
r_0 = distance_LineToPoint( P0, S, C);

% r_1 = distance{ line(P1,S), C }
r_1 = distance_LineToPoint( P1, S, C);

r = max([r_0, r_1]);

% enlarge r
r = r * preprocessingParams.ROR_enlargeFactor_xy;

% %%%% Part 2: assignment of parameters %%%%
parImg.x_0 = -(r + parImg.Delta_xy/2);
parImg.y_0 = parImg.x_0;

parImg.N_x = 2*ceil( r / parImg.Delta_xy) + 1;
parImg.N_y = parImg.N_x;


%% Computation of z_0 and N_z 

x_1 = parImg.x_0 + parImg.N_x*parImg.Delta_xy;
y_1 = x_1;

R_00 = sqrt(parImg.x_0^2 + parImg.y_0^2);
R_10 = sqrt(x_1^2     + parImg.y_0^2);
R_01 = sqrt(parImg.x_0^2 + y_1^2);
R_11 = sqrt(x_1^2     + y_1^2);

R = max([R_00, R_10, R_01, R_11]);

w_1 = par.w_d0 + par.N_dw*par.Delta_dw;


z_0 = min( 	par.w_d0 * ( R - par.u_s) / (par.u_d0 - par.u_s), ...
			par.w_d0 * (-R - par.u_s) / (par.u_d0 - par.u_s));

z_1 = max( 	w_1 * ( R - par.u_s) / (par.u_d0 - par.u_s), ...
			w_1 * (-R - par.u_s) / (par.u_d0 - par.u_s));

parImg.z_0 = z_0;
parImg.N_z = ceil(  (z_1-z_0)/(parImg.Delta_z)  );


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ROI parameters

R_roi = min([r_0, r_1]) - parImg.Delta_xy;

w_0_roi = par.w_d0 + par.Delta_dw;
w_1_roi = w_0_roi + (par.N_dw-2)*par.Delta_dw;

z_min_roi = max(	w_0_roi * (-R_roi - par.u_s) / (par.u_d0 - par.u_s), ...
					w_0_roi * ( R_roi - par.u_s) / (par.u_d0 - par.u_s));

z_max_roi = min(	w_1_roi * (-R_roi - par.u_s) / (par.u_d0 - par.u_s), ...
					w_1_roi * ( R_roi - par.u_s) / (par.u_d0 - par.u_s));

N_x_roi = 2*floor(R_roi / parImg.Delta_xy) + 1;
N_y_roi = N_x_roi;

parImg.j_xstart_roi = (parImg.N_x - N_x_roi) / 2;
parImg.j_ystart_roi = parImg.j_xstart_roi;

parImg.j_xstop_roi = parImg.j_xstart_roi + N_x_roi - 1;
parImg.j_ystop_roi = parImg.j_xstop_roi;

parImg.j_zstart_roi = round((z_min_roi - parImg.z_0) / parImg.Delta_z);
parImg.j_zstop_roi = parImg.j_zstart_roi + round((z_max_roi-z_min_roi) / parImg.Delta_z);



end



