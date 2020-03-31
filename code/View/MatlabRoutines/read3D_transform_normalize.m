function [ x, x_norm, lo, hi ] = read3D_transform_normalize( binaryPath, opts )

%% read and process data
[~, ~, ext] = fileparts(binaryPath);

dataTypeOut = 'single';
if(strcmp(ext, '.wght'))
    % x = read3D(binaryPath, 'uint8');
    x = read3D(binaryPath, 'float32', dataTypeOut);
else
	x = read3D(binaryPath, 'float32', dataTypeOut);
end

%% Permute from file type
if(strcmpi(ext, '.recon') || strcmpi(ext, '.proxMapInput'))
    x = permute(x, [3, 2, 1]);
end

%% Permute
x = permute(x, opts.indexOrder);    

%% Do flips
if(opts.flipVect(1) == 1)
	x = flip(x, 1);
end
if(opts.flipVect(2) == 1)
	x = flip(x, 2);
end
if(opts.flipVect(3) == 1)
	x = flip(x, 3);
end


%% Normalize
[lo, hi] = findBounds(x, opts.mode, opts.prctileSubsampleFactor, opts.target_lo, opts.target_hi );
x_norm = normalize01Bounds( x, lo, hi );



end

