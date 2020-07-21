function images = read_ObjScans(objectScan_folderPath, indexList)

%cd(objectScan_folderPath);

extensions = {'.tif'};
DirStructure = findBinaryFilesInFolder( objectScan_folderPath, extensions );

tiffname = DirStructure(1).name;
tempImg_path = [objectScan_folderPath '/' tiffname]
tempImg =  double(imread(tempImg_path));

images = zeros(size(tempImg,1), size(tempImg,2), length(indexList));

for i=1:length(indexList)
	
	indexFull = indexList(i);
    
	tiffname = DirStructure(indexFull+1).name;
    images(:,:,i) = read_scan( [objectScan_folderPath '/' tiffname] );
    
end
	

