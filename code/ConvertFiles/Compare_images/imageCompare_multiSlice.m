function [profileImg_list, params] = imageCompare_multiSlice(params)

% render profile image for multiple 4D volumes

[numSlices, numRecons] = size(params.fNames)

profileImg_list = cell(numRecons,1);
for c_Recon=1:numRecons
	profileImg_list{c_Recon} = [];
end


params_atomic = params;
for c_Slice=1:numSlices

	params_atomic.fNames = params.fNames(c_Slice,:);
	[profile_list, params_atomic] = imageCompare(params_atomic);

	for c_Recon=1:numRecons
		profile_single = profile_list{c_Recon};
		profileImg_list{c_Recon} = [profileImg_list{c_Recon} profile_single(:)];
	end

end

saveFolder_profileImg = [params.saveFolder, '/', 'profileImg'];
createFolder_purge(saveFolder_profileImg);

figure;
for c_Recon=1:numRecons
	f = gcf;
	clf(f);
	imagesc(profileImg_list{c_Recon}); colormap('gray');
	set(gca,'position',[0 0 1 1],'units','normalized');

	filename_imgs_tif = [saveFolder_profileImg, '/', params.imgNames{c_Recon}, '.tif'];
	filename_imgs_png = [saveFolder_profileImg, '/', params.imgNames{c_Recon}, '.png'];
	filename_imgs_fig = [saveFolder_profileImg, '/', params.imgNames{c_Recon}, '.fig'];

	imwrite(profileImg_list{c_Recon}, filename_imgs_tif);
	imwrite(profileImg_list{c_Recon}, filename_imgs_png);
	saveas(f, filename_imgs_fig);

end

close all
disp('vol compare done')

return