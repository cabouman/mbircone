function [ ] = readScans_and_preprocess( masterFile, plainParamsFile )
disp('*****************************************************************************************************')

data = readScans(masterFile, plainParamsFile );
disp('*****************************************************************************************************')

preprocess(data, masterFile, plainParamsFile );
disp('*****************************************************************************************************')

end

