function [ indices ] = arrayIndicesFromLinearIndex( linearIndex, Ns )

temp = linearIndex-1;
indices = zeros(length(Ns), 1);
for i = 1:length(Ns)
   
    indices(i) = mod(temp, Ns(i));
    temp = (temp-indices(i))/Ns(i);
    
    
end

indices = indices + 1;


end

