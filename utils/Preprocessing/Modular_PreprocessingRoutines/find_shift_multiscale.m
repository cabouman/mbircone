function [shift1, shift2] = find_shift_multiscale( scan1, scan2, shift_searchRadius, shift_numPointsGrid, shift_gridReductionRatio, shift_gridSize )

radius = shift_searchRadius;
stepsize = 2*radius/shift_numPointsGrid;
shift1 = 0;
shift2 = 0;

while( stepsize * shift_gridReductionRatio > shift_gridSize )
	[shift1, shift2] = find_shift( scan1, scan2, shift1, shift2, radius, stepsize );
	radius = radius / shift_gridReductionRatio ;
	stepsize = stepsize / shift_gridReductionRatio ;
end

return