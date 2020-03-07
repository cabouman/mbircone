#include "../CLibraries/io4d.h"
#include "../CLibraries/MBIRModularUtilities.h"
#include "recon4D.h"

int main(int argc, char *argv[])
{
	
    struct CmdLine cmdLine;
    struct PathNames pathNames;
    struct Image img;
	struct Params params;

	float normVal;


    /**
     * 		Process Command Line argument(s)
     */
    readCmdLine(argc, argv, &cmdLine);
	strcpy(pathNames.masterFile, cmdLine.masterFile);
	strcpy(pathNames.plainParamsFile, cmdLine.plainParamsFile);
	printCmdLine(&cmdLine);
	printf("Read Command Line Args ###########################################################\n\n");

	
	/**
	 * 		Read input text files
	 */
	readBinaryFNames(pathNames.masterFile, pathNames.plainParamsFile, &pathNames);
	printf("Read Binary Paths ###########################################################\n\n");
	printPathNames(&pathNames);

	readImageParamsFromBinaryFile(&pathNames, &img.params);
	printImgParams(&img.params);

	readParams(pathNames.masterFile, pathNames.plainParamsFile, &params);
	printParams(&params);
	printf("Read Params ###########################################################\n\n");

	/**
	 * 		Allocate space for images
	 */
	allocateImageData( &img );
	printf("Allocated Images ###########################################################\n\n");

	
    /**
     *      Image Initialization
     */
    readImageData( &pathNames, &img );
    printf("Initialized Images ###########################################################\n\n");

    /*debug*/
    // if(DEBUG_FLAG) initializeDenoisedImg_scalar(&img, 0);
    if(DEBUG_FLAG) initializeDenoisedImg_fromNoisy( &img );

    /*debug*/
    if(DEBUG_FLAG) normVal = computeRMS( img.denoised , img.params.N_t, img.params.N_x, img.params.N_y, img.params.N_z);
    if(DEBUG_FLAG) printf("denoised: RMS = %f \n",normVal );
    if(DEBUG_FLAG) normVal = computeRMS( img.noisy , img.params.N_t, img.params.N_x, img.params.N_y, img.params.N_z);
    if(DEBUG_FLAG) printf("noisy: RMS = %f \n",normVal );
    /*printf("pot = %f\n",QGGMRFPotential( 20.0, &params) );*/

	#if 1
	/**
	 * 		Reconstruction
	 */
	MBIR_4D(&img, &params, &pathNames);
	printf("Denoised Images ###########################################################\n\n");
	#endif

	/**
	 * 		Write Image Files
	 */
	writeImageData( &pathNames, &img );
	printf("Wrote Images to disk ###########################################################\n\n");

	/**
	 * 		Free allocated data
	 */
    free_pathName(&pathNames);
	free_imageData(&img);
    printf("Deallocated Images ###########################################################\n\n");


	return 0;
}
