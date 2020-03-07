function [ x_chunked, lb1, ub1, lb2, ub2, lb3, ub3 ] = split3DArray(x, haloRadius, maxChunkSize)

for i = 1:3
    [ lb{i}, ub{i} ] = limits_chunk( size(x,i), maxChunkSize, haloRadius );
    numChunks{i} = length(lb{i});
end


for i = 1:3
    numChunks{i};
end

i = 1;
for i1 = 1:numChunks{1}
    for i2 = 1:numChunks{2}
        for i3 = 1:numChunks{3}

            lb1(i) = lb{1}(i1);
            lb2(i) = lb{2}(i2);
            lb3(i) = lb{3}(i3);
            
            ub1(i) = ub{1}(i1);
            ub2(i) = ub{2}(i2);
            ub3(i) = ub{3}(i3);
            
            x_chunked{i} = x(lb1(i):ub1(i), lb2(i):ub2(i), lb3(i):ub3(i));
            
            i = i+1;
            
        end
    end
end

end