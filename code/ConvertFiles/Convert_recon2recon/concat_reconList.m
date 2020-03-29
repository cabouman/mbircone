function [] = concat_recon( fileList_in, catAxis, padLen, padIntensity, fName_out )

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Includes (note: paths relative to function location)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mfilepath=fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../../Modular_MatlabRoutines'));
addpath(fullfile(mfilepath,'../MatlabRoutines'));

x_cat = getRecon(fileList_in{1});

padLens = [size(x_cat,1), size(x_cat,2), size(x_cat,3)];
padLens(catAxis) = padLen;
padRecon = padIntensity * ones( padLens );


for i=2:length(fileList_in)

	x = getRecon(fileList_in{i});

	x_cat = cat(catAxis, x_cat, padRecon, x);

end


%% Permute from file type
[~, ~, ext] = fileparts(fName_out);
if(strcmpi(ext, '.recon') || strcmpi(ext, '.proxMapInput'))
    x_cat = permute(x_cat, [3, 2, 1]);
end
write3D(fName_out, x_cat, 'float32');


return

function x = getRecon(fName_in)

	[~, ~, ext] = fileparts(fName_in);
	x = read3D(fName_in, 'float32');
	%% Permute from file type
	if(strcmpi(ext, '.recon') || strcmpi(ext, '.proxMapInput'))
	    x = permute(x, [3, 2, 1]);
	end

return