function [ x ] = readAndScale_GEVolFile( fName_vol, rawPCRData )


N1 = rawPCRData.N1;
N2 = rawPCRData.N2;
N3 = rawPCRData.N3;
Minimum = rawPCRData.Minimum;
Maximum = rawPCRData.Maximum;

fid = fopen(fName_vol);
x = fread(fid, N1*N2*N3, 'uint16');
fclose(fid);

x = reshape(x, [N1, N2, N3]);
x = permute(x, [3, 2, 1]);

% take values from 0...UINTMAX (integer) to [Minimum, Maximum] (float)
x = double(x);
x = normalize01(x);
x = (x * (Maximum - Minimum)) + Minimum;
x = single(x);

end

