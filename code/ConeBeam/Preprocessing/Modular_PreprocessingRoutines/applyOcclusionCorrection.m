function sino = applyOcclusionCorrection( sino, occlusion_sino, preprocessingParams, binaryFNames )

switch preprocessingParams.occusionCorrectionType

case 0
	% do nothing

case 1
	fName_sino_occlusion = getOcclusionPath(binaryFNames.sino)
	if(exist(fName_sino_occlusion, 'file') ~= 0)
	    sino_occlusion = read3D( fName_sino_occlusion, 'float32');
	    sino = sino - sino_occlusion;
	else
		error(['applyOcclusionCorrection: file ', fName_sino_occlusion, ' does not exist']);
	end

case 2
	% sino = sino - occlusion_sino;
	fName_sino_occlusion = getOcclusionPath(binaryFNames.sino);
	write3D(fName_sino_occlusion, occlusion_sino, 'float32');

otherwise
	error('applyOcclusionCorrection: unrecongnized occusionCorrectionType');

end

return



function fName_sino_occlusion = getOcclusionPath(sinoPath)

[pathstr,name,ext] = fileparts(sinoPath);
subFolderName = '/FOV_correction/';
outFolder = [ pathstr, subFolderName ];
fName_sino_occlusion = [outFolder, name, '.occlusion', ext];

return