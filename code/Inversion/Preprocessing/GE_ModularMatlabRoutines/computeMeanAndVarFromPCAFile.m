function [ meanImg, varImg, rawPCAData ] = computeMeanAndVarFromPCAFile( fName, preprocessingParams )


rawPCAData = readPCAFile( fName );
numSamples = preprocessingParams.N_avg;

if(rawPCAData.NumberImages < numSamples)
    error(['Error: Number of images required (', num2str(numSamples) ') < Number of images available (', num2str(rawPCAData.NumberImages), ') | fName = ' fName]);
end

sum1 = zeros(rawPCAData.DimY, rawPCAData.DimX);
sum2 = zeros(rawPCAData.DimY, rawPCAData.DimX);

TIFFPathNames = readTIFFPathNamesFromPCAFile( fName );

for i = 1:numSamples
    temp = double(imread(TIFFPathNames{i}));
    sum1 = sum1 + temp;
    sum2 = sum2 + temp.^2;
end

meanImg = sum1 / numSamples;
varImg = sum2 / (numSamples - 1) - meanImg.^2;



end

