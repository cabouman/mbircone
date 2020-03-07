function path_abs = getAbsPath_withNull(path_ref, path_rel)
% converts non-empty relative path to absolute paths
% empty relative paths convert to empty absolute paths

if ~isempty(path_rel)
	path_abs = [path_ref path_rel];
	% absolutePath_relativeTo has problems with spaces in path
	% path_abs = absolutePath_relativeTo( path_rel, path_ref);
else
	path_abs = '' ;
end

return