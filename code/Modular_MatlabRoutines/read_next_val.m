function val = read_next_val(filepath, tag, section)

% Searches a .nsipro file with the given tag in the section
% Returns the string next to the tag


tag_str = [ '<' tag '>' ];
section_start = [ '<' section '>' ];
section_end = [ '</' section '>' ];

fp = fopen(filepath);
if fp==-1
	error('read_next_val: Wrong File Path: %s', filepath);
end

C = textread(filepath, '%s','delimiter', '\n');

section_start_ind = zeros(1,length(C));
section_end_ind = zeros(1,length(C));
for i=1:length(C)
	section_start_ind(i) = double(startsWith_(C{i},section_start));
	section_end_ind(i) = double(startsWith_(C{i},section_end));
end

in_section = 0;
line_found = -1;
for i=1:length(C)
	if section_start_ind(i)==1
		in_section = 1;
	end
	if section_end_ind(i)==1
		in_section = 0;
	end
	
	if startsWith_(C{i},tag_str) && in_section==1
		line_found = i;
		break;
	end
end


if line_found==-1
	val = [];
else
	val = strrep(C{line_found},tag_str,'');
end

end


