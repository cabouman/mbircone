function [ TIFFPathNames ] = readTIFFPathNamesFromPCAFile( fName )

[pathStr, fName, ~] = fileparts(fName);


TIFFPathNames = glob([pathStr, '/', fName, '[0-9][0-9][0-9][0-9][0-9].tif']);



end

