function [ ] = view3D_folder( folderName_binary, opts)

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Includes (note: paths relative to function location)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mfilepath=fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../../Modular_MatlabRoutines'));
addpath(fullfile(mfilepath,'../MatlabRoutines'));
addpath(fullfile(mfilepath,'MatlabRoutines'));

%%
extensions = {'.sino', '.wght', '.recon', '.scan'};
[ DirStructure ] = findBinaryFilesInFolder( folderName_binary, extensions );

%%

if(isempty(DirStructure))
	warning('WARNING: Folder does not contain any binary files')
end

	

for i = 1:length(DirStructure)
    binaryPath = [folderName_binary, '/', DirStructure(i).name];

    
    view3D_single(binaryPath, opts);
    
end

end



