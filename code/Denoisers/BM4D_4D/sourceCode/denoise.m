function [] = denoise(masterFile, plainParamsFile)

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  Includes (note: paths relative to function location)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
mfilepath=fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../ModularMatlabCode/'));
addpath(fullfile(mfilepath,'../../../Modular_MatlabRoutines/'));
addpath(fullfile(mfilepath,'../../ModularMatlabCode_4D/'));
addpath(fullfile(mfilepath,'BM4D_v3p2/'));

par = readDenoisingParams(masterFile, plainParamsFile);
[ fileLists ] = readImageFileLists_denoiser_4D( masterFile, plainParamsFile );

if(par.verbose==1)
	printStruct(par, 'par');
	fileLists
end


x = read4D(fileLists.noisyImageNames, 'float32');

permute_vect = [ 1 2 3 4 ];
const_dims = par.const_dims_4D;

for i=1:length(const_dims)
	permute_vect = set_const_dims(permute_vect, const_dims, i);
end

x = permute(x, permute_vect);

switch length(const_dims)
	case 1
		for i=1:size(x,1)
			temp_img = x(i,:,:,:);
			temp_img = shiftdim(temp_img,1);

			temp_img2 = denoise_2D_3D( temp_img, '3D', par);
			x(i,:,:,:) = temp_img2;
		end
	case 2
		for i=1:size(x,1)
			for j=1:size(x,2)
				temp_img = x(i,j,:,:);
				temp_img = shiftdim(temp_img,2);
				% disp(size(temp_img))

				temp_img2 = denoise_2D_3D( temp_img, '2D', par);
				x(i,j,:,:) = temp_img2;
				
			end
		end
	otherwise
		error('denoise bm4d: wrong number of elements in const_dims')
end


x = permute(x, permute_vect);

write4D(fileLists.denoisedImageNames, x, 'float32');


end

function permute_vect_new = set_const_dims(permute_vect, const_dims, index)

permute_vect_new = permute_vect;
permute_vect_new(const_dims(index)) = index;
permute_vect_new(index) = const_dims(index);

end