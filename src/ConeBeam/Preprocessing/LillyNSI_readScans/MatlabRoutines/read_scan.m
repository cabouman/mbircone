function scanImage = read_scan(scanPath)


scanImage = imread( scanPath );

dataClass = class(scanImage);

if ~isempty(strfind(dataClass,'int'))
	% integer class
	scanImage = double(scanImage)/double(intmax(dataClass));
end