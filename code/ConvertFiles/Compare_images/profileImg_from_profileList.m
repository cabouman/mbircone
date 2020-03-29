function [profileImg] = profileImg_from_profileList(profileList)

profileImg = [];

for i=1:length(profileList)
	profileVect = profileList{i};
	profileImg = [ profileImg profileVect(:) ];

end

return