function [] = volRender_reconList(fNameList, opts)

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Includes (note: paths relative to function location)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mfilepath=fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../../Modular_MatlabRoutines'));
addpath(fullfile(mfilepath,'../MatlabRoutines'));
addpath(fullfile(mfilepath,'../Convert_recon2recon'));

binaryPath_1 = fNameList{1};
[binaryDir, baseName, ext] = fileparts(binaryPath_1);
viewOutputFolder = [binaryDir '/', opts.folderSuffix];
dispName = [baseName, ext];

for i=1:length(fNameList)

	[~, baseName, ext] = fileparts(fNameList{i});
	img = read3D(fNameList{i}, 'float32');
	%% Permute from file type
	if(strcmpi(ext, '.recon') || strcmpi(ext, '.proxMapInput'))
	    img = permute(img, [3, 2, 1]);
	end

	img = orient_recon(img, opts);
	[img, ~, ~, ~] = crop3DArray(img, opts.range1, opts.range2, opts.range3);
	[lo, hi] = findBounds(img, opts.mode, opts.prctileSubsampleFactor, 0, 1);
	
	img_norm = normalize01Bounds(img, lo, hi);


	f = figure(1);
	img_norm = flip(img_norm, 3);


	% img_norm = imresize3(img_norm, 2);

	Alphamap = linspace(0,1,256)';
	if isfield(opts, 'transparencyLims')
		transparencyLims = normalize01Bounds(opts.transparencyLims, lo, hi);

		Alphamap = (Alphamap-transparencyLims(1))/(transparencyLims(2)-transparencyLims(1));
		Alphamap(Alphamap<0) = 0;
		Alphamap(Alphamap>1) = 1;
	end

	volshow(img_norm, 'BackgroundColor', opts.BackgroundColor, 'ScaleFactors', opts.ScaleFactors, ...
		'CameraPosition', opts.CameraPosition, 'CameraTarget', opts.CameraTarget, ...
		'Isovalue', opts.Isovalue, 'Renderer', opts.Renderer, 'IsosurfaceColor', opts.surfaceColor, ...
		'Alphamap', Alphamap);
		
	
	I = getframe(gcf);
	img = rgb2gray(I.cdata);
	img = double(img)/256;

	img_toSave(:,:,i) = img;

end

createFolder_purge(viewOutputFolder);

generateOptsFile( viewOutputFolder, dispName, opts);

generateGif( viewOutputFolder, dispName, img_toSave );
generateMPEG4Video( viewOutputFolder, dispName, img_toSave );
generateSingleImages( viewOutputFolder, dispName, img_toSave, 'tif');
generateSingleImages( viewOutputFolder, dispName, img_toSave, 'png');

return
