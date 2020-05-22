function [ s ] = generateOptsFile( viewOutputFolder, dispName, opts)

str = evalc('printStruct(opts, ''opts'', 1) ');

fName = [viewOutputFolder, '/', dispName, '_opts.txt'];

fid = fopen(fName,'w');
fprintf(fid, '%s', str);
fclose(fid);

end

