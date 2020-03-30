function [ angleList ] = readViewAngleList( masterFile, plainParamsFile )

executablePath = plainParamsFile;
get_set = 'get';
masterField = 'viewAngleList';
value = '';
resolveFlag = '-r';

fName = plainParams(executablePath, get_set, masterFile, masterField, '', value, resolveFlag);

C = textread(fName, '%s','delimiter', '\n');
numFiles = str2num(C{1});
angleList = zeros(numFiles, 1);


% Read individual filenames
for i=1:(numFiles)  
    % Read individual 3D vols 
    angleList(i) = str2num(C{i+1});
end


end

