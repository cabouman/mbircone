function [ ] = readScans_and_preprocess_genScan( masterFile, plainParamsFile )
disp('*****************************************************************************************************')

data = readScans_genScan(masterFile, plainParamsFile );
disp('*****************************************************************************************************')

preprocess(data, masterFile, plainParamsFile );
disp('*****************************************************************************************************')

end

