function recon_new = imtranslate_int(recon, shiftVect)

for i=1:3
	minId = 1 + shiftVect(i);
	maxId = size(recon, i) + shiftVect(i);
	if shiftVect(i)>0
		minId_old(i) = 1 + shiftVect(i);
		maxId_old(i) = size(recon, i);
		minId_new(i) = 1;
		maxId_new(i) = size(recon, i)-shiftVect(i);
	else
		minId_new(i) = 1 - shiftVect(i);
		maxId_new(i) = size(recon, i);
		minId_old(i) = 1;
		maxId_old(i) = size(recon, i)+shiftVect(i);
	end
end
% minId_new
% maxId_new
% minId_old
% maxId_old

recon_new = 0*recon;
recon_new(minId_new(1):maxId_new(1),minId_new(2):maxId_new(2),minId_new(3):maxId_new(3)) = recon(minId_old(1):maxId_old(1),minId_old(2):maxId_old(2),minId_old(3):maxId_old(3));

return