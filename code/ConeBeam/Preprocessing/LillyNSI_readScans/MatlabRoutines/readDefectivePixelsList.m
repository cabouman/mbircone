function [ defectivePixelList ] = readDefectivePixelsList( defects_filePath )

section_lines = read_section(defects_filePath, 'Defective Pixels');
num_defectivePixels = length(section_lines);
fprintf('Number of defective pixels: %d\n',num_defectivePixels);

defectivePixelList = zeros( num_defectivePixels, 2);

for i=1:num_defectivePixels
	coord = str2num(section_lines{i});
	defectivePixelList(i,:) = [ coord(1) coord(2) ];
end


return