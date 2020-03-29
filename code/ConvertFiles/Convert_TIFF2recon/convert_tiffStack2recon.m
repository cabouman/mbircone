function [ ] = convert_tiffStack2recon(tiffStackPath, reconPath)
% takes .tiff stack file as an input and writes .recon file to disc in the same location

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Includes (note: paths relative to function location)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mfilepath=fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../../Modular_MatlabRoutines'));

extensions = {'.tif'};
DirStructure = findBinaryFilesInFolder( tiffStackPath, extensions );
numTIF = length(DirStructure);

tiffname = DirStructure(1).name;
tempImg =  double(imread([tiffStackPath '/' tiffname]));

disp('Reading images ...');
images = zeros(size(tempImg,1), size(tempImg,2), numTIF);
for i=1:numTIF
    tiffname = DirStructure(i).name;
    images(:,:,i) = double(imread( [tiffStackPath '/' tiffname] ));
end


images = images / max(images(:));

% permute for recon format
images = permute(images, [3 2 1]);

% Save file to disc
disp(['Writing to ', reconPath, ' ...']);
write3D(reconPath, images, 'float32');

