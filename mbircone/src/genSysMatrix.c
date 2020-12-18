#include <stdio.h>

#include "io3d.h"
#include "MBIRModularUtilities3D.h"
#include "computeSysMatrix.h"



int main(int argc, char *argv[])
{
    struct CmdLine cmdLine;
    struct PathNames pathNames;
    struct SinoParams sinoParams;
    struct ImageParams imgParams;
	struct ReconParams reconParams;
    struct ViewAngleList viewAngleList;
    
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
	printCmdLine(&cmdLine);


	/**
	 * 		Read input text files
	 */
	readBinaryFNames(pathNames.masterFile, pathNames.plainParamsFile, &pathNames);
  	printPathNames(&pathNames);

	readSinoParams(pathNames.masterFile, pathNames.plainParamsFile, &sinoParams);
	printSinoParams(&sinoParams);

	readImageParams(pathNames.masterFile, pathNames.plainParamsFile, &imgParams);
	printImgParams(&imgParams);

	readReconParams(pathNames.masterFile, pathNames.plainParamsFile, &reconParams);
	computeSecondaryReconParams(&reconParams, &imgParams);
	printReconParams(&reconParams);

	readAndAllocateViewAngleList(pathNames.masterFile, pathNames.plainParamsFile, &viewAngleList, &sinoParams);


	if (reconParams.verbosity>=3)
	{
		printPathNames(&pathNames);
		printSinoParams(&sinoParams);
		printImgParams(&imgParams);
		printReconParams(&reconParams);
		
	}
	/**
	 * 		System Matrix stuff
	 */
	computeSysMatrix(&sinoParams, &imgParams, &A, &reconParams, &viewAngleList);
	printSysMatrixParams(&A);
	writeSysMatrix(pathNames.sysMatrix, &sinoParams, &imgParams, &A);


	/**
	 * 		Freeing space
	 */
	freeSysMatrix(&A);
	freeViewAngleList(&viewAngleList);

	return 0;
}
