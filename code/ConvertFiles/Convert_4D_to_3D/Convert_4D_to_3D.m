function [] = Convert_4D_to_3D(fNameList, optsList)

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Includes (note: paths relative to function location)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mfilepath=fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../../Modular_MatlabRoutines'));
addpath(fullfile(mfilepath,'../MatlabRoutines'));
addpath(fullfile(mfilepath,'../Convert_recon2recon'));

for k=1:length(optsList)

	opts = optsList{k}
	binaryPath_1 = fNameList{1};
	[binaryDir, baseName, ext] = fileparts(binaryPath_1);
	viewOutputFolder = [binaryDir '/', opts.folderSuffix];
	dispName = [baseName, ext];

	createFolder_purge(viewOutputFolder);

end

for i=1:length(fNameList)

	[~, baseName, ext] = fileparts(fNameList{i});
	vol_orig = read3D(fNameList{i}, 'float32');
	%% Permute from file type
	if(strcmpi(ext, '.recon') || strcmpi(ext, '.proxMapInput'))
	    vol_orig = permute(vol_orig, [3, 2, 1]);
	end

	for k=1:length(optsList)

		img = getImg(vol_orig, optsList{k});

		img_toSave{k}(:,:,i) = img;
	end

end



for k=1:length(optsList)

	opts = optsList{k}

	if(strcmpi(ext, '.recon') || strcmpi(ext, '.proxMapInput'))
	    img_tmp = permute(img_toSave{k}, [3, 2, 1]);
	end

	fName_out = [viewOutputFolder, '/', opts.name, num2str(opts.sliceDim), ext]
	write3D(fName_out, img_tmp, 'float32');

	generateOptsFile( viewOutputFolder, [opts.name, num2str(opts.sliceDim), ext, '_convert'], opts);

end



return


function img = getImg(vol_orig, opts)

	if opts.sliceId==-1
		opts.sliceId = round(size(vol_orig,opts.sliceDim)/2);
	end

	switch opts.sliceDim
	case 1
		img = vol_orig(opts.sliceId,:,:);
		img = squeeze(img);
		img = permute(img, [2 1]);

	case 2 
		img = vol_orig(:,opts.sliceId,:);
		img = squeeze(img);
		img = permute(img, [2 1]);

	case 3
		img = vol_orig(:,:,opts.sliceId);

	otherwise
		error('subSelectionVol: wrong coord value')
	end

	img = permute(img, opts.permuteSlice);

	if(opts.flipSlice(1) == 1)
		img = flip(img, 1);
	end

	if(opts.flipSlice(2) == 1)
		img = flip(img, 2);
	end
	

return
