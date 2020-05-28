function [ ] = writeViewAngleList( angleList, masterFile, plainParamsFile )

executablePath = plainParamsFile;
get_set = 'get';
masterField = 'viewAngleList';
value = '';
resolveFlag = '-r';

fName = plainParams(executablePath, get_set, masterFile, masterField, '', value, resolveFlag);

fid = fopen(fName, 'w');


fprintf(fid, '%d\n', length(angleList));
for i = 1:length(angleList)
    fprintf(fid, '%f\n', angleList(i));
end

fclose(fid);


end

