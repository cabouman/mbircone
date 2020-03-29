function scale_recon(vol_out, vol1_fName, vol2_fName, opts1, opts2, patchParams)
% scales vol1 to same scale as vol2
% each pixel in vol2 =y, vol1 x, y = ax+b. find a & b

[ x1, x1_norm, lo1, hi1 ] = read3D_transform_normalize( vol1_fName, opts1 );

[ x2, x2_norm, lo2, hi2 ] = read3D_transform_normalize( vol2_fName, opts2 );


vol1_dark_patch = x1( patchParams.dark_vol1min(1):patchParams.dark_vol1max(1),  patchParams.dark_vol1min(2):patchParams.dark_vol1max(2),  patchParams.dark_vol1min(3):patchParams.dark_vol1max(3) );
vol2_dark_patch = x2( patchParams.dark_vol2min(1):patchParams.dark_vol2max(1),  patchParams.dark_vol2min(2):patchParams.dark_vol2max(2),  patchParams.dark_vol2min(3):patchParams.dark_vol2max(3) );

vol1_bright_patch = x1( patchParams.bright_vol1min(1):patchParams.bright_vol1max(1),  patchParams.bright_vol1min(2):patchParams.bright_vol1max(2),  patchParams.bright_vol1min(3):patchParams.bright_vol1max(3) );
vol2_bright_patch = x2( patchParams.bright_vol2min(1):patchParams.bright_vol2max(1),  patchParams.bright_vol2min(2):patchParams.bright_vol2max(2),  patchParams.bright_vol2min(3):patchParams.bright_vol2max(3) );


clearvars x2

vol1_dark_val = mean(vol1_dark_patch(:));
vol2_dark_val = mean(vol2_dark_patch(:));
vol1_bright_val = mean(vol1_bright_patch(:));
vol2_bright_val = mean(vol2_bright_patch(:));


figure,
subplot(2,1,1);
plot(vol1_dark_patch(:));
hold on;
plot(vol1_bright_patch(:));
legend('dark','bright');

subplot(2,1,2);
plot(vol2_dark_patch(:));
hold on;
plot(vol2_bright_patch(:));
legend('dark','bright');

a = ( vol2_bright_val - vol2_dark_val )/( vol1_bright_val - vol1_dark_val );
b = vol2_bright_val - a * vol1_bright_val ; 

x1_new = x1;
for i=1:size(x1,1)
	for j=1:size(x1,2)
		for k=1:size(x1,3)
			x1_new(i,j,k) = a * x1(i,j,k) + b;
			if x1_new(i,j,k) < 0 
				x1_new(i,j,k) = 0;
			end
		end
	end
end

x1_new = permute(x1_new, opts2.indexOrder);
write3D(vol_out, x1_new, 'float32');

end

