function [ fName_new ] = prependAppendFileName( prefix, fName, suffix)


[PATHSTR,NAME,EXT] = fileparts(fName);

fName_new = [PATHSTR, '/', prefix, NAME, suffix,EXT];


end

