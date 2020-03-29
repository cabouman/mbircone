function x = convert_NSI2Recon( fName_body, fName_head, fName_recon )

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Includes (note: paths relative to function location)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mfilepath=fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../../Modular_MatlabRoutines'));

sizes = str2num(read_next_val(fName_head, 'Voxels', 'North Star Imaging Volume Header'));

fid = fopen(fName_body, 'r');
x = fread(fid, 'single');
fclose(fid);

x = reshape(x, sizes(3), sizes(1), sizes(2));

if(exist('fName_recon', 'var'))
    write3D(fName_recon, x, 'float32');
end




end

