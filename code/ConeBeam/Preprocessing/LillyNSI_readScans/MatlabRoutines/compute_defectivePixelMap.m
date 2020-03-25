function [ defectivePixelMap ] = compute_defectivePixelMap( defectivePixelList, zeroImage )

numdefectivePixels = length(defectivePixelList);
defectivePixelMap = zeroImage;

for i=1:numdefectivePixels
	index_x = defectivePixelList(i,1);
	index_y = defectivePixelList(i,2);

	defectivePixelMap(index_y,index_x) = 1;
end

return