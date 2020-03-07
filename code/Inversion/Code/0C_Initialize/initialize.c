#include "../0A_CLibraries/io3d.h"
#include "../0A_CLibraries/MBIRModularUtilities3D.h"




int main(int argc, char *argv[])
{
    struct CmdLine cmdLine;
    struct PathNames pathNames;
    struct Sino sino;
    struct ImageF img;
	struct ReconParams reconParams;
	struct SysMatrix A;

	/**
	 * 		Reset Log files
	 */
	resetFile(LOG_TIME);
	resetFile(LOG_PROGRESS);

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


	if (reconParams.verbosity>=2)
	{
		printPathNames(&pathNames);
		printSinoParams(&sino.params);
		printImgParams(&img.params);
		printReconParams(&reconParams);
		printSysMatrixParams(&A);
		
	}

	/**
	 * 		Read System Matrix
	 */
	readSysMatrix(pathNames.sysMatrix, &sino.params, &img.params, &A);
	/*printSysMatrixParams(&A);*/

	/**
	 * 		Parallel stuff
	 */
	omp_set_num_threads(reconParams.numThreads);

	/**
	 * 		Allocate
	 */
	sino.vox = (float***) allocateSinoData3DCone(&sino.params, sizeof(float));



    if (reconParams.isForwardProjectPhantom && reconParams.isPhantomPresent)
    {
    	printf("Forward Projecting ...\n");
    	/**
    	 * 		Allocate
    	 */
		img.phantom = (float***) allocateImageFData3DCone( &img.params, sizeof(float), 0);
	    
		/**
		 * 		Initialize
		 */
	    readImageFData3DCone(pathNames.phantom, (void***)img.phantom, &img.params, 0, "float");
		
	    setFloatArray2Value(&sino.vox[0][0][0], sino.params.N_beta*sino.params.N_dv*sino.params.N_dw, 0);
	    forwardProject3DCone( sino.vox, img.phantom, &img.params, &A, &sino.params);

	    /**
	     * 		Write
	     */
		writeSinoData3DCone(pathNames.sino, (void***)sino.vox, &sino.params, "float");

		/**
		 * 		Free
		 */
	    mem_free_3D((void***)img.phantom);
		
    }
    else if (reconParams.isRecomputeWeight)
    {
    	printf("Reconpute Weight Recon Error ...\n");
		sino.wgt = (float***) allocateSinoData3DCone(&sino.params, sizeof(float));
		img.wghtRecon = (float***) allocateImageFData3DCone( &img.params, sizeof(float), 0);

		readSinoData3DCone(pathNames.sino, (void***)sino.vox, &sino.params, "float");

		initializeWght(&A, &sino);
		initializeWghtRecon(&A, &sino, &img, &reconParams);

		writeSinoData3DCone(pathNames.wght, (void***)sino.wgt, &sino.params, "float");
		writeImageFData3DCone(pathNames.wghtRecon, (void***)img.wghtRecon, &img.params, 0, "float");

	    mem_free_3D((void***)sino.wgt);
	    mem_free_3D((void***)img.wghtRecon);

    }
    else
    {
    	printf("Initializing Error ...\n");
    	/**
    	 * 		Allocate
    	 */
		sino.e = (float***) allocateSinoData3DCone(&sino.params, sizeof(float));
		sino.wgt = (float***) allocateSinoData3DCone(&sino.params, sizeof(float));
		sino.mask = (char***) allocateSinoData3DCone(&sino.params, sizeof(char));
		img.vox = (float***) allocateImageFData3DCone( &img.params, sizeof(float), 0);
		img.vox_roi = (float***) allocateImageFData3DCone( &img.params, sizeof(float), 1);
		img.wghtRecon = (float***) allocateImageFData3DCone( &img.params, sizeof(float), 0);
		img.proxMapInput = (float***) allocateImageFData3DCone( &img.params, sizeof(float), 0);
		img.lastChange = (float***) mem_alloc_3D(img.params.N_x, img.params.N_y, reconParams.numZiplines, sizeof(float));
		img.timeToChange = (unsigned char***) mem_alloc_3D(img.params.N_x, img.params.N_y, reconParams.numZiplines, sizeof(unsigned char));
		

		/**
		 * 		Initialize
		 */
		readSinoData3DCone(pathNames.sino, (void***)sino.vox, &sino.params, "float");
		readSinoData3DCone(pathNames.wght, (void***)sino.wgt, &sino.params, "float");
		if (reconParams.isUsePhantomToInitErrSino && reconParams.isPhantomPresent)
		{
			readImageFData3DCone(pathNames.phantom, (void***)img.vox, &img.params, 0, "float");
		}
		else
		{
		    setImageF2Value_insideMask(img.vox, &img.params, reconParams.InitVal_recon);
		}
	    setImageF2Value_insideMask(img.proxMapInput, &img.params, reconParams.InitVal_proxMapInput);
	    setFloatArray2Value(&img.lastChange[0][0][0], img.params.N_x*img.params.N_y*reconParams.numZiplines, 0.0);
	    setUCharArray2Value(&img.timeToChange[0][0][0], img.params.N_x*img.params.N_y*reconParams.numZiplines, 0);
		copyImageF2ROI(&img);



	    /**
	     *      Error Initialization: e = y - Ax
	     *      Sinogram Mask Initialization
	     */
	    errorInitialization(&img, &A, &sino, &reconParams);
	    initializeSinoMask(&A, &sino, &img, &reconParams);
		initializeWghtRecon(&A, &sino, &img, &reconParams);


    	
	    /**
	     * 		Write
	     */
		writeImageFData3DCone(pathNames.recon, (void***)img.vox, &img.params, 0, "float");
		writeImageFData3DCone(pathNames.wghtRecon, (void***)img.wghtRecon, &img.params, 0, "float");
		writeImageFData3DCone(pathNames.reconROI, (void***)img.vox_roi, &img.params, 1, "float");
		writeImageFData3DCone(pathNames.proxMapInput, (void***)img.proxMapInput, &img.params, 0, "float");
		write3DData(pathNames.lastChange, (void***)img.lastChange, img.params.N_x, img.params.N_y, reconParams.numZiplines, "float");
		write3DData(pathNames.timeToChange, (void***)img.timeToChange, img.params.N_x, img.params.N_y, reconParams.numZiplines, "unsigned char");

		writeSinoData3DCone(pathNames.errsino, (void***)sino.e, &sino.params, "float");
		writeSinoData3DCone(pathNames.sinoMask, (void***)sino.mask, &sino.params, "char");
	    
		/**
		 * 		Free
		 */
	    mem_free_3D((void***)sino.e);
	    mem_free_3D((void***)sino.wgt);
	    mem_free_3D((void***)sino.mask);

	    mem_free_3D((void***)img.vox);
	    mem_free_3D((void***)img.vox_roi);
	    mem_free_3D((void***)img.wghtRecon);
	    mem_free_3D((void***)img.proxMapInput);
	    mem_free_3D((void***)img.lastChange);
	    mem_free_3D((void***)img.timeToChange);
    }



    /**
     * 		Free
     */
    mem_free_3D((void***)sino.vox);
    freeSysMatrix(&A);


	return 0;
}
