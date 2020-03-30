function [outList] = process_reconList( fNameList_in, opts, dir_out )

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Includes (note: paths relative to function location)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mfilepath=fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../../Modular_MatlabRoutines'));
addpath(fullfile(mfilepath,'../MatlabRoutines'));

for i=1:length(fNameList_in)

	fName_in = fNameList_in{i};
	[~, baseName, ext] = fileparts(fName_in);
	fName_out = [dir_out, '/', baseName, ext];

	x = read3D(fName_in, 'float32');
	%% Permute from file type
	if(strcmpi(ext, '.recon') || strcmpi(ext, '.proxMapInput'))
	    x = permute(x, [3, 2, 1]);
	end

	% process
	switch opts.description

	case 'crop'
		x = crop_recon(x, opts);

	case 'orient'
		x = orient_recon(x, opts);

	case 'normalize'
		x = normalize_recon(x, opts);

	otherwise
		error('process_reconList: invalid description');

	end

	if(exist('dir_out', 'var'))
		%% Permute from file type
		if(strcmpi(ext, '.recon') || strcmpi(ext, '.proxMapInput'))
		    x = permute(x, [3, 2, 1]);
		end
	    write3D(fName_out, x, 'float32');
	    outList{i} = fName_out;
	else
		outList{i} = x;
	end

end

if(exist('dir_out', 'var'))
	generateOptsFile( dir_out, opts.description, opts);
end


return

