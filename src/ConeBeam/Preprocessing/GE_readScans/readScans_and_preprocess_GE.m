function [ ] = readScans_and_preprocess_GE( masterFile, plainParamsFile )

data = readScans_GE(masterFile, plainParamsFile );

preprocess(data, masterFile, plainParamsFile );

end

