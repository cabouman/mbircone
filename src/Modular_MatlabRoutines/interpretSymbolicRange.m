function [ range ] = interpretSymbolicRange(img, dim, symrange)


countingNumbers = 1:size(img,dim);
range = eval(['countingNumbers(round(' symrange '))']);

end