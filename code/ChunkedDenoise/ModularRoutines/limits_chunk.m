function [ lowerLimit, upperLimit ] = limits_chunk( numEntries, maxChunkSize, haloSize )
% chunkSize = blockSize + 2 * haloSize


if(haloSize >= maxChunkSize/2)
    error('Choose haloSize < maxChunkSize/2');
end

if(haloSize >= numEntries/2)
    warning('haloSize < numEntries/2');
    haloSize = 0;
    disp('Automatically chooses haloSize = 0');
end

numEntries_temp = numEntries - 2*haloSize;
maxBlockSize = maxChunkSize - 2*haloSize;

numBlocks = ceil(numEntries/maxBlockSize);
avgBlockLength = numEntries_temp/numBlocks;

index = 0:numBlocks(1)-1;

lowerLimit = floor(index * avgBlockLength) + 1 - haloSize;
lowerLimit = max(lowerLimit, 1);


upperLimit = floor((index+1) * avgBlockLength) + haloSize;
upperLimit = min(upperLimit, numEntries_temp);

lowerLimit(1) = lowerLimit(1) - haloSize;
upperLimit(end) = upperLimit(end) + haloSize;

lowerLimit = lowerLimit + haloSize;
upperLimit = upperLimit + haloSize;



end

