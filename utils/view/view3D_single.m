function [ ] = view3D_single(binaryPath, opts)

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Includes (note: paths relative to function location)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mfilepath=fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../misc_routines_matlab'));
addpath(fullfile(mfilepath,'utils'));
%addpath(fullfile(mfilepath,'MatlabRoutines'));

%%

% opts.mode:                            'absolute a b' or 'percentile a b' | (a, b are numbers)
% opts.prctileSubsampleFactor = D       1, 2, ... uses x(1:D:end) for finding percentiles (faster)
% opts.isGenerateMP4:                   0 or 1 
% opts.isFramesJPG:                     0 or 1 
% opts.isFramesTIF:                     0 or 1 
% opts.isGenerateGIF                    0 or 1
% opts.isGenerateColorbarEnvironment    0 or 1

% opts.indexOrder:        [permutation(1 2 3)] or don't set this field (then determined by file extension) 
% opts.flip:              0: do nothing, 1: flip 180 degrees (similar to time reversel)
% opts.rotate:            rotates in display plane. (90 degrees * opts.rotate)

% opts.figurePrintSize:   1 or 0.5 or 2 or such
% opts.figureDispSize:    1 or 0.5 or 2 or such
% opts.relFontSize:       1 or 0.5 or 2 or such

% opts.range_1			
% opts.range_2
% opts.range_3

% opts.folderSuffix  		must be nonempty

%%
if(length(opts.folderSuffix) == 0)
	error('opts.folderSuffix must be nonempty');
end

%%
[ img, img_norm, lo, hi ] = read3D_transform_normalize( binaryPath, opts );


for i = 1:2

	%% Names of folder and figures
	[~, basename_woe, ext] = fileparts(binaryPath);
	dispName = [basename_woe, ext];


	if(i == 1) % Complete volume
	    viewOutputFolder = [binaryPath, opts.folderSuffix, ''];
	    stats.range1 = 1:size(img,1);
	    stats.range2 = 1:size(img,2);
	    stats.range3 = 1:size(img,3);
	end

	if(i == 2) % Here the cropping happens
	    viewOutputFolder = [binaryPath, opts.folderSuffix, '_cropped'];

		[img, stats.range1, stats.range2, stats.range3] = crop3DArray(img, opts.range1, opts.range2, opts.range3);
		[img_norm, stats.range1, stats.range2, stats.range3] = crop3DArray(img_norm, opts.range1, opts.range2, opts.range3);

		if(length(img) == 0)
			warning('Specified ranges results in empty array. No output generated.');
            continue;
		end

	end

	%% Create/Purgue Output Folder
    createFolder_purge(viewOutputFolder);

	%% Generate Histogram
	figureNumber = 1;
	numBins = 200;
    description = '(full)';
	generateHistogram( viewOutputFolder, dispName, img, [min(img(:)), max(img(:))], figureNumber, numBins, description);

    figureNumber = 1;
    description = '(window)';
    temp_img = img; temp_img(img<=lo)=lo; temp_img(img>=hi) = hi;
	generateHistogram( viewOutputFolder, dispName, temp_img, [lo, hi], figureNumber, numBins, description);
    clear temp_img;
        
	%% generate stats text file
	stats.min_norm = lo;
	stats.max_norm = hi;
	generateStatsFile( viewOutputFolder, dispName, img, stats);
    
    %%
    generateOptsFile( viewOutputFolder, dispName, opts);


	%% Generate Gif File
	if(opts.isGenerateGIF)
	    generateGif( viewOutputFolder, dispName, img_norm );
    end
    
	%% Write sinlge frames into a folder
	if(opts.isFramesJPG)
	    generateSingleImages( viewOutputFolder, dispName, img_norm, 'jpg');
	end
	if(opts.isFramesTIF)
	    generateSingleImages( viewOutputFolder, dispName, img_norm, 'tif');
	end
	if(opts.isFramesPNG)
	    generateSingleImages( viewOutputFolder, dispName, img_norm, 'png');
	end

	%% Generate environment with title and colorbar for the video
	figureNumber = 2;
	if(i == 1)
		generateColorbarEnvironment( viewOutputFolder, dispName, img(:,:,ceil(end/2)), lo, hi, figureNumber, opts);
	end
	if(i == 2)
		if(opts.isGenerateColorbarEnvironment)
			generateColorbarEnvironment( viewOutputFolder, dispName, img, lo, hi, figureNumber, opts);
		else
			generateColorbarEnvironment( viewOutputFolder, dispName, img(:,:,1), lo, hi, figureNumber, opts);
		end
	end	

	%% Generate video (MPEG-4)
	if(opts.isGenerateMP4)
	    generateMPEG4Video( viewOutputFolder, dispName, img_norm );
    end
    
    %% Generate projections across 3 canonical planes
    generateCanonicalProjections( viewOutputFolder, dispName, img, opts)

end



return;



end
