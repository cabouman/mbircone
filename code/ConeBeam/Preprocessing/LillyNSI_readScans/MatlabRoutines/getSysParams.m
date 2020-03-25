function sysParams = getSysParams(dataSetInfo)

%source to det dist
val = read_next_val(dataSetInfo.nsipro_filePath,'source','Result');
val = str2num(val);
sysParams.u_s = val(3);

% First Detector Pixel
val = read_next_val(dataSetInfo.nsipro_filePath,'reference','Result');
val = str2num(val);
sysParams.v_d1  = val(1); 
sysParams.w_d1  = val(2); 
sysParams.u_d1  = val(3);

% Detector Pixel Spacing
val = read_next_val(dataSetInfo.nsipro_filePath,'pitch','Object Radiograph');
val = str2num(val);
sysParams.Delta_dv = val(1);
sysParams.Delta_dw = val(2);

% Detector Size
val = read_next_val(dataSetInfo.nsipro_filePath,'width pixels','Detector');
val = str2num(val);
sysParams.N_dv = val;

val = read_next_val(dataSetInfo.nsipro_filePath,'height pixels','Detector');
val = str2num(val);
sysParams.N_dw = val;


% Coordinate of center of rotation 
sysParams.u_r = 0;
sysParams.v_r = 0;

% Number of Views
val = read_next_val(dataSetInfo.nsipro_filePath,'number','Object Radiograph');
val = str2num(val);
sysParams.numAcquiredScans = val;


sysParams.TotalAngle = str2num( read_next_val(dataSetInfo.nsipro_filePath,'Rotation range','CT Project Configuration') ); 



