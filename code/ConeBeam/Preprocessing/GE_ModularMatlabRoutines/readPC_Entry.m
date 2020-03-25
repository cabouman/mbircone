function [ value ] = readPC_Entry(value, sectionNameIs, sectionNameShould, varNameShould, line)
    
if(startsWith_(line, varNameShould) & strcmp(sectionNameIs, sectionNameShould))
	splitline = strsplit(line, '=');
	value = str2num(splitline{2});
end


end

