function fName_new = genBinaryFileName_single( fName, suffix, id ) 

[filepath, name, ext] = fileparts(fName);

% fprintf('%s %s %s %d\n',filepath, name, ext, id );

name_new = [ name suffix '_' num2str(id) ];

fName_new = [ filepath '/' name_new ext ];

return