function indexList = computeIndexList(preprocessingParams, numAcquiredScans, TotalAngle)

% with this option set N_beta_all to -1 and it will take all the acquired scans
if(preprocessingParams.N_beta_all==-1)
    preprocessingParams.N_beta_all = numAcquiredScans;
end

N_beta_all = preprocessingParams.N_beta_all;


num_timePoints_all = preprocessingParams.num_timePoints_all;
index_timePoints_all = preprocessingParams.index_timePoints_all;

num_viewSubsets = preprocessingParams.num_viewSubsets ;
index_viewSubsets = preprocessingParams.index_viewSubsets ;


if index_timePoints_all >= num_timePoints_all
	error('computeIndexList: index_timePoints_all exceeds limit');
end

if index_viewSubsets >= num_viewSubsets
	error('computeIndexList: index_timePoints_all exceeds limit');
end


indexList_acquiredScans = 0:numAcquiredScans-1;

% --------------- Lossy sampling -----------------------------------------------------
% arbirary+lossy selection
subset_acquiredScans = interpretSymbolicRange(indexList_acquiredScans(:), 1, preprocessingParams.subset_acquiredScans);
indexList_1 = indexList_acquiredScans(subset_acquiredScans);

% lossy selection of N_beta_all views
indexList_2 = subsample_array(indexList_1, N_beta_all);

% --------------- Conserving sampling ------------------------------------------------
% Conserving selection of Time point
indexList_3 = select_contiguousSubset(indexList_2, num_timePoints_all, index_timePoints_all);

% Conserving seleciton of view subset
indexList_4 = select_interlacedSubset(indexList_3, num_viewSubsets, index_viewSubsets);
% ------------------------------------------------------------------------------------

indexList = indexList_4;

return



function array_sampled = subsample_array(array, newSize)

if newSize>length(array)
	error('subsample_array: new size must be smaller');
end

index_sampled = floor( length(array) * [0:newSize-1]/newSize ) + 1;

array_sampled = array(index_sampled);

return