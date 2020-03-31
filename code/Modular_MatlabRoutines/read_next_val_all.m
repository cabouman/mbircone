function valList = read_next_val_all(filepath, tag, section)

% Searches a .nsipro file with the given tag in the section
% Returns the string next to the tag
% for multiple matches returns all values


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
valList = {};
for i=1:length(C)
	if section_start_ind(i)==1
		in_section = 1;
	end
	if section_end_ind(i)==1
		in_section = 0;
	end
	
	if startsWith_(C{i},tag_str) && in_section==1
		val = strrep(C{i},tag_str,'');
		valList = {valList{:}, val};
	end
end


end


