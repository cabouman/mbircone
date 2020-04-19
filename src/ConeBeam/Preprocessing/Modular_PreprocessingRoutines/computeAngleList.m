function angleList = computeAngleList( indexList, numAcquiredScans, TotalAngle, rotationDirection );
% in radians

switch rotationDirection
	case 1
		angleList = mod( (2*pi/360) * TotalAngle * indexList / numAcquiredScans , 2*pi );
	case 0
		angleList = mod( -(2*pi/360) * TotalAngle * indexList / numAcquiredScans , 2*pi );
	otherwise
		error('computeAngleList: invalid rotationDirection');
end


return
