function [ ] = write3D( fName, img , dataType)


sizes = [size(img,1) size(img,2) size(img,3)];

fid = fopen(fName, 'w');

if(fid == -1)
	error(['Cannot open file ', fName]);
end

%flip sizes to match C notation (row major)
sizes = sizes(end:-1:1);

fwrite(fid, sizes, 'int32');
fwrite(fid, img, dataType);

fclose(fid);

end

