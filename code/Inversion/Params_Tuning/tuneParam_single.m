function [ shift_new, error_list_all, s_list_all, error_scan] = tuneParam_single( paramTuningParams, shift, sino, forwardProj, wght )

% subsample views
new_sino = eval( [ 'sino(' paramTuningParams.view_list ')'] );
new_forwardProj = eval( [ 'forwardProj(' paramTuningParams.view_list ')'] );
new_wght = eval( [ 'wght(' paramTuningParams.view_list ')'] );


error_list_all = [];
s_list_all = [];
switch paramTuningParams.searchMethod.name

case 'multiscaleSearch'
	% [shift1, shift2] = find_shift_multiscale( new_sino, new_forwardProj, paramTuningParams.shift_searchRadius, paramTuningParams.shift_numPointsGrid, paramTuningParams.shift_gridReductionRatio, paramTuningParams.shift_gridSize );

	radius = paramTuningParams.searchMethod.shift_searchRadius;
	stepsize = 2*radius/paramTuningParams.searchMethod.shift_numPointsGrid;
	shift_gridReductionRatio = paramTuningParams.searchMethod.shift_gridReductionRatio;
	shift_searchRadius = paramTuningParams.searchMethod.shift_searchRadius;
	shift_gridSize = paramTuningParams.searchMethod.shift_gridSize;

	while( stepsize > shift_gridSize )
		[shift, error_list, s_list, error_scan] = find_shift_x( new_sino, new_forwardProj, new_wght, shift, radius, stepsize );
		error_list_all = [error_list_all; error_list];
		s_list_all = [s_list_all; s_list];
		radius = radius / shift_gridReductionRatio ;
		stepsize = stepsize / shift_gridReductionRatio ;
	end

otherwise
	error('tuneParams: paramTuningParams.searchMethod not valid');
end

[~, I] = sort(s_list_all);
s_list_all = s_list_all(I);
error_list_all = error_list_all(I);

shift_new = shift;

