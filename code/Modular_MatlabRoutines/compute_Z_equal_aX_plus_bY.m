function compute_Z_equal_aX_plus_bY(fNameList_Z, a, fNameList_X, b, fNameList_Y, dataType)
% fNameList's can be cell of strings or single string

if mean(size(fNameList_Z)==size(fNameList_X)) ~= 1
	error(sprintf('compute_Z_equal_aX_plus_bY: size mismatch fNameList_Z(%d) and fNameList_X(%d)\n',size(fNameList_Z),size(fNameList_X)));
end
if mean(size(fNameList_Z)==size(fNameList_Y)) ~= 1
	error(sprintf('compute_Z_equal_aX_plus_bY: size mismatch fNameList_Z(%d) and fNameList_Y(%d)\n',size(fNameList_Z),size(fNameList_Y)));
end

fNameList_Z = makeCell(fNameList_Z);
fNameList_X = makeCell(fNameList_X);
fNameList_Y = makeCell(fNameList_Y);

for i = 1:length(fNameList_Z)

	z = a * read3D(fNameList_X{i}, dataType) + b * read3D(fNameList_Y{i}, dataType);
	write3D(fNameList_Z{i}, z, dataType);
end


end