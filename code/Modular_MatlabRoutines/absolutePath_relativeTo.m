function fName_abs = absolutePath_relativeTo(fName_rel, base_file_or_folder)

mfilepath=fileparts(which(mfilename));
addpath(mfilepath);
backup = pwd;

% find base
if(exist(base_file_or_folder, 'dir')==7)
	folder = base_file_or_folder;
elseif(exist(base_file_or_folder, 'file') == 2)
	folder = fileparts(base_file_or_folder);
else
	error('base_file_or_folder does not exist')
end


% go to base and find abs path
cd(folder);
fName_abs = GetFullPath(fName_rel);
cd(backup);
