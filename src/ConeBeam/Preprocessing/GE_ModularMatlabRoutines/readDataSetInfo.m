function [ dataSetInfo ] = readDataSetInfo( masterFile, plainParamsFile )

executablePath = plainParamsFile;
get_set = 'get';
masterField = 'dataSetInfo';
value = '';

dataSetInfo.mode = str2num(plainParams(executablePath, get_set, masterFile, masterField, 'mode', value, ''));

dataSetInfo.pcaObject = plainParams(executablePath, get_set, masterFile, masterField, 'pcaObject', value, '-r');

if(dataSetInfo.mode ~= 0)

	dataSetInfo.pcaDark = plainParams(executablePath, get_set, masterFile, masterField, 'pcaDark', value, '-r');
	dataSetInfo.pcaBlank = plainParams(executablePath, get_set, masterFile, masterField, 'pcaBlank', value, '-r');

end

end


