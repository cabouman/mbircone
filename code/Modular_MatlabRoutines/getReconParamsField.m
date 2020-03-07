function [ field ] = getReconParamsField(fieldString, masterPathNamesFName)

masterPathNames = readMasterPathNames(masterPathNamesFName);
reconParams = readReconParams(masterPathNames.reconParams);

eval(['field = reconParams.', fieldString, ';']);


end