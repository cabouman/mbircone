function [fName_List] = get_Inversion_FNameList( invMasterList, plainParamsFile, binaryField )

% Add plainParams.m file to path
d = fileparts(GetFullPath(plainParamsFile));
addpath(d);

executablePath = plainParamsFile;
get_set = 'get';
value = '';
resolveFlag = '-r';

numVols_inversion = length(invMasterList);
fName_List = cell(1,numVols_inversion);
for t=1:numVols_inversion
	masterFile = invMasterList{t};
	fName_List{t} = plainParams(executablePath, get_set, masterFile, 'binaryFNames', binaryField, value, resolveFlag);
end

return
