function [ data ] = mirror_image( data, mirrorDirection )

switch(mirrorDirection)
case 'horz'
	data = data(:,end:-1:1);
case 'vert'
	data = data(end:-1:1,:);
end