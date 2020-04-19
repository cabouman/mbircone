function [ ] = GE_convertVolData(fName_vol)
% takes .vol file as an input and writes .recon file to disc in the same location

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Includes (note: paths relative to function location)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mfilepath=fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../GE_ModularMatlabRoutines'));
addpath(fullfile(mfilepath,'../../../Modular_MatlabRoutines'));

%% File names (assumes .pcr file is in the same folder and has the same file name as .vol file)
[ path, filename, ext] = fileparts(fName_vol);
fName_pcr = fullfile(path, [filename, '.pcr']);
fName_out = fullfile(path, [filename, '.recon']);

%% pcr file
rawPCRData  = readPCRFile( fName_pcr );

%% read vol file and scale it
x = readAndScale_GEVolFile( fName_vol, rawPCRData );


%% store file to disc
disp(['Writing to ', fName_out, ' ...']);
write3D(fName_out, x, 'float32');

end