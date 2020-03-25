#include "../0A_CLibraries/io3d.h"
#include "../0A_CLibraries/MBIRModularUtilities3D.h"
#include "recon3DCone.h"




int main(int argc, char *argv[])
{
    struct CmdLine cmdLine;
    struct PathNames pathNames;
    struct Sino sino;
    struct ImageF img;
	struct ReconParams reconParams;
	struct SysMatrix A;


    /**
     * 		Process Command Line argument(s)
     */
    readCmdLine(argc, argv, &cmdLine);
	strcpy(pathNames.masterFile, cmdLine.masterFile);
	strcpy(pathNames.plainParamsFile, cmdLine.plainParamsFile);
	


	
	/**
	 * 		Read input text files
	 */
	readBinaryFNames(pathNames.masterFile, pathNames.plainParamsFile, &pathNames);

	readSinoParams(pathNames.masterFile, pathNames.plainParamsFile, &sino.params);

	readImageFParams(pathNames.masterFile, pathNames.plainParamsFile, &img.params);

	readReconParams(pathNames.masterFile, pathNames.plainParamsFile, &reconParams);
	computeSecondaryReconParams(&reconParams, &img.params);

	/**
	 * 		Read System Matrix
	 */
	readSysMatrix(pathNames.sysMatrix, &sino.params, &img.params, &A);


	if (reconParams.verbosity>=2)
	{
		printPathNames(&pathNames);
		printSinoParams(&sino.params);
		printImgParams(&img.params);
		printReconParams(&reconParams);
		printSysMatrixParams(&A);
		
	}
	/**
	 * 		Allocate space for sinogram
	 */
	sino.wgt = (WEIGHTDATATYPE***) allocateSinoData3DCone(&sino.params, sizeof(WEIGHTDATATYPE));
	sino.e = (float***) allocateSinoData3DCone(&sino.params, sizeof(float));
	sino.vox = (float***) allocateSinoData3DCone(&sino.params, sizeof(float));
	sino.mask = (char***) allocateSinoData3DCone(&sino.params, sizeof(char));

	/**
	 * 		Allocate space for image
	 */
	img.vox = (float***) allocateImageFData3DCone( &img.params, sizeof(float), 0);
	img.vox_roi = (float***) allocateImageFData3DCone( &img.params, sizeof(float), 1);
	img.wghtRecon = (float***) allocateImageFData3DCone( &img.params, sizeof(float), 0);
	img.proxMapInput = (float***) allocateImageFData3DCone( &img.params, sizeof(float), 0);
	img.lastChange = (float***) mem_alloc_3D(img.params.N_x, img.params.N_y, reconParams.numZiplines, sizeof(float));
	img.timeToChange = (unsigned char***) mem_alloc_3D(img.params.N_x, img.params.N_y, reconParams.numZiplines, sizeof(unsigned char));
	if(reconParams.isPhantomPresent)
		img.phantom = (float***) allocateImageFData3DCone( &img.params, sizeof(float), 0);

	/*
	 *		Read sinogram data
	 */
	readSinoData3DCone(pathNames.wght, (void***)sino.wgt, &sino.params, WEIGHTDATATYPE_string);
	readSinoData3DCone(pathNames.errsino, (void***)sino.e, &sino.params, "float");
	readSinoData3DCone(pathNames.sino, (void***)sino.vox, &sino.params, "float");
	readSinoData3DCone(pathNames.sinoMask, (void***)sino.mask, &sino.params, "char");
	
    /**
     *      Image Initialization
     */
    readImageFData3DCone(pathNames.recon, (void***)img.vox, &img.params, 0, "float");
    readImageFData3DCone(pathNames.wghtRecon, (void***)img.wghtRecon, &img.params, 0, "float");
    readImageFData3DCone(pathNames.proxMapInput, (void***)img.proxMapInput, &img.params, 0, "float");
    read3DData(pathNames.lastChange, (void***)img.lastChange, img.params.N_x, img.params.N_y, reconParams.numZiplines, "float");
    read3DData(pathNames.timeToChange, (void***)img.timeToChange, img.params.N_x, img.params.N_y, reconParams.numZiplines, "unsigned char");

	if(reconParams.isPhantomPresent)
	    readImageFData3DCone(pathNames.phantom, (void***)img.phantom, &img.params, 0, "float");




	/**
	 * 		Reconstruction
	 */
	MBIR3DCone(&img, &sino, &reconParams, &A, &pathNames);

	/**
	 * 		Write Image Files
	 */
	writeImageFData3DCone(pathNames.recon, (void***)img.vox, &img.params, 0, "float");
	writeImageFData3DCone(pathNames.reconROI, (void***)img.vox_roi, &img.params, 1, "float");

	/**
	 * 		Write Sinogram files
	 */
	writeSinoData3DCone(pathNames.errsino, (void***)sino.e, &sino.params, "float");

	/**
	 * 		Free allocated data
	 */
    mem_free_3D((void***)img.vox);
    mem_free_3D((void***)img.proxMapInput);
    mem_free_3D((void***)img.vox_roi);
    mem_free_3D((void***)img.lastChange);
    mem_free_3D((void***)img.timeToChange);
    if(reconParams.isPhantomPresent)
	    mem_free_3D((void***)img.phantom);

    mem_free_3D((void***)sino.wgt);
    mem_free_3D((void***)sino.e);
    mem_free_3D((void***)sino.mask);

    freeSysMatrix(&A);


	return 0;
}
