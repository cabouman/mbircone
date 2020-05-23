function [ ] = readScans_and_preprocess_LillyNSI( masterFile, plainParamsFile )

data = readScans_LillyNSI(masterFile, plainParamsFile );

preprocess(data, masterFile, plainParamsFile );

end
