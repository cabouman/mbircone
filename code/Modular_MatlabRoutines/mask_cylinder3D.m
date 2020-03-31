function x_m = mask_cylinder3D(x, cent, radiusList)

x_m = x;
for i3=1:size(x,3)
	for i1=1:size(x,1)
	    for i2=1:size(x,2)
	        
	        i = [ i1 i2 ];
	        if norm(i-cent)>radiusList(i3)
	            x_m(i1,i2,i3) = 0;
	        else
	        	x_m(i1,i2,i3) = 1;
	        end
	        
	    end
	end
end


return