function x = convert_NSI2Recon_multiFile( fName_header, fName_recon )

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Includes (note: paths relative to function location)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mfilepath=fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../../Modular_MatlabRoutines'));

sizes = str2num(read_next_val(fName_header, 'Voxels', 'North Star Imaging Volume Header'));

fNameList_data = read_next_val_all(fName_header, 'Name', 'Files')
sizeList_data = read_next_val_all(fName_header, 'NbSlices', 'Files')

dirpath_header = fileparts(fName_header);

sumSize = 0;
for i=1:length(sizeList_data)

	path_data = fullfile(dirpath_header, fNameList_data{i});
	fid = fopen(path_data, 'r');
	x = fread(fid, 'single');
	fclose(fid);

	size_data = str2num(sizeList_data{i});

	x = reshape(x, sizes(3), sizes(1), size_data);
	sumSize = sumSize + size_data;

	volList{i} = x;

end

assert(sumSize==sizes(2), 'Sum of sizes do not match header size');

recon = cat(3, volList{:});

if(exist('fName_recon', 'var'))
    write3D(fName_recon, recon, 'float32');
end




end

