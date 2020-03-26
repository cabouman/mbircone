function [x] = averageInversionRecons(plainParamsFile, inversionMasterList_fName)

inversionMasterList = readNDFileList(inversionMasterList_fName);

numVols = length(inversionMasterList);


for i = 1:numVols
	
    binaryFNames = readBinaryFNames( inversionMasterList{i}, plainParamsFile );
    x_single = read3D(binaryFNames.recon, 'float32');
    if(i==1)
        x = x_single;
    else
        x = x + x_single;
    end
end
x = x / numVols;


end
