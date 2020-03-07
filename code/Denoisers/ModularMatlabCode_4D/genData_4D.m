function [] = genData_4D(masterFile, plainParamsFile)

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Includes (note: paths relative to function location)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mfilepath=fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../../Modular_MatlabRoutines/'));

[ fRoots ] = readFileRoots_denoiser_4D( masterFile, plainParamsFile );
% [ fileLists ] = readImageFileLists_denoiser_4D( masterFile, plainParamsFile );

fRoots
% fileLists

disp('Generating Phantom . . .');
tic
% t z y x
% [~, noisy] = generateNoisyDummy4D([60 357 357 115], 0.2, 0.05);
[img] = generateDummy4D([10 20 100 100], 0.2, );
[noisy_img] = addNoise(img, 0.05)
% N=10;
% [~, noisy] = generateNoisyDummy4D([N N N N], 10, 2);
disp('Image size:');
disp(size(noisy_img));

disp('Writing Images . . .');
noisyFname_list = generate_4DfNamesList( fRoots.noisyImageFNameRoot,'4D_replica_', '', 1, size(noisy_img,1) );
writeFileList( fRoots.noisyBinaryFName_timeList, noisyFname_list );
write4D(noisyFname_list, noisy_img, 'float32');

denoisedFname_list = generate_4DfNamesList(fRoots.denoisedImageFNameRoot,'4D_replica_', '', 1, size(noisy_img,1));
writeFileList( fRoots.denoisedBinaryFName_timeList, denoisedFname_list );
write4D(denoisedFname_list, noisy_img, 'float32');

timeGen=toc;

fprintf('Time to generate data: %d secs\n',timeGen);

return


function [img] = generateDummy4D(sizeVect, level)
% generate Dummy Data 

N1 = sizeVect(1);
N2 = sizeVect(2);
N3 = sizeVect(3);
N4 = sizeVect(4);

N = N1*N2*N3*N4;
maxN = max([N1 N2 N3 N4]);
Rad = round(0.3*maxN);

img = zeros(N1,N2,N3,N4);
center1 = round(N1/2);
center2 = round(N2/2);
center3 = round(N3/2);
center4 = round(N4/2);
center = [center1 center2 center3 center4];
for i1=1:N1
	for i2=1:N2
		for i3=1:N3
			for i4=1:N4
				id = [ i1 i2 i3 i4];
				if norm(center(:)-id(:)) < Rad
					img(i1,i2,i3,i4) = level;
				else
					img(i1,i2,i3,i4) = 0;
				end
			end
		end
	end
end

return

function [noisy_img] = addNoise(img, noiseSigma)

rng('default');

N1 = size(img,1);
N2 = size(img,2);
N3 = size(img,3);
N4 = size(img,4);

noise = noiseSigma*randn(N1,N2,N3,N4);
noisy_img = img + noise;

return