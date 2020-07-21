function [ ] = readScans_and_preprocess_NSI( masterFile, plainParamsFile )

data = readScans_NSI(masterFile, plainParamsFile );

preprocess(data, masterFile, plainParamsFile );

end
