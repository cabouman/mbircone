
nsipro_filePath = ''

%source to det dist
val = read_next_val(nsipro_filePath,'source','Result');
val = str2num(val);
sysParams.u_s = val(3);

% First Detector Pixel
val = read_next_val(nsipro_filePath,'reference','Result');
val = str2num(val);
sysParams.u_d1  = val(3);
sysParams.v_d1  = val(1);
sysParams.w_d1  = val(2);

% Detector Pixel Spacing
val = read_next_val(nsipro_filePath,'pitch','Object Radiograph');
val = str2num(val);
sysParams.Delta_dv = val(1);
sysParams.Delta_dw = val(2);

% Detector Size
val = read_next_val(nsipro_filePath,'width pixels','Detector');
val = str2num(val);
sysParams.N_dv = val;

val = read_next_val(nsipro_filePath,'height pixels','Detector');
val = str2num(val);
sysParams.N_dw = val;

% Number of Views
val = read_next_val(nsipro_filePath,'number','Object Radiograph');
val = str2num(val);
sysParams.numAcquiredScans = val;


sysParams.TotalAngle = str2num( read_next_val(nsipro_filePath,'Rotation range','CT Project Configuration') );

sysParams



