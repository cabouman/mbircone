function [ w ] = triangleND( N123 )

if(length(N123) == 1)
    w = triangle1D(N123);
else
    N12 = N123(1:end-1);
    N3 = N123(end);
    
    w12 = triangleND(N12);
    w3 = triangle1D(N3);
    
    w = zeros(N123);
    
    colons = repmat(':, ', 1, length(N12));
    
    for i3 = 1:N3
        cmd = [ 'w(', colons, 'i3) = w12 * w3(i3);'];
        eval(cmd);
    end
    
    
end

end


function w = triangle1D(N)

n = 0:N-1;
w = (N+1)/2 - abs(n-(N-1)/2);

end

