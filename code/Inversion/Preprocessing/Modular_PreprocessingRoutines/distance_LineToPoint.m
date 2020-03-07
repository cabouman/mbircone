function [ dist ] = distance_LineToPoint( A, B, P )
% Computes the distance between (the line throught the points A and B) and (the point P). 

x1 = A(1);
y1 = A(2);

x2 = B(1);
y2 = B(2);

x0 = P(1);
y0 = P(2);

num = abs(...
    (y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1 ...
          );
      
denom = sqrt(...
    (y2-y1)^2 + (x2-x1)^2 ...
             );

dist = num/denom;
end

