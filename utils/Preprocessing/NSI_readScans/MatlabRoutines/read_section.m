function section_lines = read_section(filepath, section)

% Searches a .nsipro file with the given tag in the section
% Returns the string next to the tag


section_start = [ '<' section '>' ];
section_end = [ '</' section '>' ];

fp = fopen(filepath);
if fp==-1
	error('read_next_val: Wrong File Path: %s', filepath);
end

C = textread(filepath, '%s','delimiter', '\n');

section_start_ind = 1;
section_end_ind = length(C);
for i=1:length(C)
	% C{i}
	if startsWith_(C{i},section_start) 
		section_start_ind = i;
	end
	if startsWith_(C{i},section_end) 
		section_end_ind = i;
	end
end

% section_start_ind
% section_end_ind

section_index = (section_start_ind+1):(section_end_ind-1);
section_lines = C(section_index);

return


