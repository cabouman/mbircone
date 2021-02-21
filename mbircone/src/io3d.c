
#include <libgen.h>
#include <unistd.h>
#include "io3d.h"
#include "plainParams.h"

/* Read Command-line */
void readCmdLine(int argc, char *argv[], struct CmdLine *cmdLine)
{
	char ch;

	

	cmdLine->masterFile[0] = '\0';
	cmdLine->plainParamsFile[0] = '\0';
	cmdLine->modes[0] = '\0';

	/* get options */
	while ((ch = getopt(argc, argv, "a:b:c:")) != EOF)
	{

		switch (ch)
		{
			case 'a':
			{
				sprintf(cmdLine->masterFile, "%s", optarg);
				break;
			}
			case 'b':
			{
				sprintf(cmdLine->plainParamsFile, "%s", optarg);
				break;
			}			
			case 'c':
			{
				sprintf(cmdLine->modes, "%s", optarg);
				break;
			}			
			default:
			{
				printf("\nError in readCmdLine: Argument \"-%c\" not recognized\n", ch);
				exit(-1);
				break;
			}
		}
	}
	if(cmdLine->masterFile[0] == '\0')
	{
		printf("\nError in readCmdLine: masterFile not specified\n");
		exit(-1);
	}
	if(cmdLine->plainParamsFile[0] == '\0')
	{
		printf("\nError in readCmdLine: plainParamsFile not specified\n");
		exit(-1);
	}
	if(cmdLine->modes[0] == '\0')
	{
		printf("\nError in readCmdLine: modes not specified\n");
		exit(-1);
	}
}

void printCmdLine(struct CmdLine *cmdLine)
{
	printf("\n"
			"Command line arguments read:\n"
			"\tmasterFile = %s\n"
			"\tplainParamsFile = %s\n"
			"\tmodes = %s\n"
			"\n",\
			cmdLine->masterFile, cmdLine->plainParamsFile, cmdLine->modes);					
}

void readLineFromFile(char *fName, char lineNr, char *line, int lineSize)
{
	/**
	 * 		Returns the specified line. Result is in "line". Newline character removed.
	 */
   	long int i, reachedEOF;
   	FILE *fid;


	fid = fopen(fName, "r");
    if (fid == NULL)
    {
        fprintf(stderr, "ERROR in readLineFromFile: Failed to open file \"%s\"", line);
        exit(-1);
    }

    reachedEOF = 0;
    for(i = 0; i < lineNr ; ++i)
    {
        if(0 == fgets(line, lineSize, fid))
        {
            reachedEOF = 1;
            break;
        }
    }
    if(reachedEOF)
    {
        fprintf(stderr, "ERROR in readLineFromFile: File \"%s\" has less than %d lines\n", fName, lineNr);
        exit(-1);
    }

    /* Remove newline character at the end of the line*/
    line[strlen(line)-1] = '\0';

    fclose(fid);

}


void absolutePath_relativeToFileLocation(char *fName_rel, char *base_file, char *fName_abs)
{
	char directory_backup[1000];
	char *dir_of_base_file;
	char buff[1000];

	getcwd(directory_backup, 1000);

	strcpy(buff, base_file); /* Protect base_file from being changed */
	dir_of_base_file = dirname(buff);
	
	chdir(dir_of_base_file);
	realpath(fName_rel, fName_abs);
	chdir(directory_backup);

}

long int str2int(char *str)
{
	long int value;
	sscanf(str, "%ld", &value);
	return value;
}

double str2double(char *str)
{
	double value;
	sscanf(str, "%lf", &value);
	return value;
}


void readBinaryFNames(char *masterFile, char *plainParamsFile, struct PathNames *pathNames)
{
	char get_set[1000] = "get";
	char masterField[1000] = "binaryFNames";
	char resolveFlag[100] = "-r";


	plainParams(plainParamsFile, get_set, masterFile, masterField, "sino", pathNames->sino, resolveFlag);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "driftSino", pathNames->driftSino, resolveFlag);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "origSino", pathNames->origSino, resolveFlag);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "wght", pathNames->wght, resolveFlag);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "errSino", pathNames->errSino, resolveFlag);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "recon", pathNames->recon, resolveFlag);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "proxMapInput", pathNames->proxMapInput, resolveFlag);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "lastChange", pathNames->lastChange, resolveFlag);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "timeToChange", pathNames->timeToChange, resolveFlag);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "sysMatrix", pathNames->sysMatrix, resolveFlag);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "wghtRecon", pathNames->wghtRecon, resolveFlag);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "projInput", pathNames->projInput, resolveFlag);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "projOutput", pathNames->projOutput, resolveFlag);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "backprojlikeInput", pathNames->backprojlikeInput, resolveFlag);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "backprojlikeOutput", pathNames->backprojlikeOutput, resolveFlag);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "estimateSino", pathNames->estimateSino, resolveFlag);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "consensusRecon", pathNames->consensusRecon, resolveFlag);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "jigMeasurementsSino", pathNames->jigMeasurementsSino, resolveFlag);

}

void readSinoParams(char *masterFile, char *plainParamsFile, struct SinoParams *sinoParams)
{

	char get_set[1000] = "get";
	char masterField[1000] = "sinoParams";
	char resolveFlag[100] = "";
	char temp[1000] = "";

	plainParams(plainParamsFile, get_set, masterFile, masterField, "N_dv", temp, resolveFlag);
	sinoParams->N_dv = str2int(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "N_dw", temp, resolveFlag);
	sinoParams->N_dw = str2int(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "N_beta", temp, resolveFlag);
	sinoParams->N_beta = str2int(temp);


	plainParams(plainParamsFile, get_set, masterFile, masterField, "Delta_dv", temp, resolveFlag);
	sinoParams->Delta_dv = str2double(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "Delta_dw", temp, resolveFlag);
	sinoParams->Delta_dw = str2double(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "u_s", temp, resolveFlag);
	sinoParams->u_s = str2double(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "u_r", temp, resolveFlag);
	sinoParams->u_r = str2double(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "v_r", temp, resolveFlag);
	sinoParams->v_r = str2double(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "u_d0", temp, resolveFlag);
	sinoParams->u_d0 = str2double(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "v_d0", temp, resolveFlag);
	sinoParams->v_d0 = str2double(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "w_d0", temp, resolveFlag);
	sinoParams->w_d0 = str2double(temp);

	/* Initialize these to avoid unpredictable behavior */
	sinoParams->weightScaler_value = -1.0;

}

void readImageParams(char *masterFile, char *plainParamsFile, struct ImageParams *imgParams)
{
	char get_set[1000] = "get";
	char masterField[1000] = "imgParams";
	char temp[1000] = "";
	char resolveFlag[100] = "";

	plainParams(plainParamsFile, get_set, masterFile, masterField, "x_0", temp, resolveFlag);
	imgParams->x_0 = str2double(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "y_0", temp, resolveFlag);
	imgParams->y_0 = str2double(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "z_0", temp, resolveFlag);
	imgParams->z_0 = str2double(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "N_x", temp, resolveFlag);
	imgParams->N_x = str2int(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "N_y", temp, resolveFlag);
	imgParams->N_y = str2int(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "N_z", temp, resolveFlag);
	imgParams->N_z = str2int(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "Delta_xy", temp, resolveFlag);
	imgParams->Delta_xy = str2double(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "Delta_z", temp, resolveFlag);
	imgParams->Delta_z = str2double(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "j_xstart_roi", temp, resolveFlag);
	imgParams->j_xstart_roi = str2int(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "j_ystart_roi", temp, resolveFlag);
	imgParams->j_ystart_roi = str2int(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "j_zstart_roi", temp, resolveFlag);
	imgParams->j_zstart_roi = str2int(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "j_xstop_roi", temp, resolveFlag);
	imgParams->j_xstop_roi = str2int(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "j_ystop_roi", temp, resolveFlag);
	imgParams->j_ystop_roi = str2int(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "j_zstop_roi", temp, resolveFlag);
	imgParams->j_zstop_roi = str2int(temp);


    imgParams->N_x_roi = imgParams->j_xstop_roi - imgParams->j_xstart_roi + 1;
    imgParams->N_y_roi = imgParams->j_ystop_roi - imgParams->j_ystart_roi + 1;
    imgParams->N_z_roi = imgParams->j_zstop_roi - imgParams->j_zstart_roi + 1;


	/* if updated, also update "printImgParams" */
}

void readReconParams(char *masterFile, char *plainParamsFile, struct ReconParams *reconParams)
{
	char get_set[1000] = "get";
	char masterField[1000] = "reconParams";
	char temp[1000] = "";
	char resolveFlag[100] = "";

	plainParams(plainParamsFile, get_set, masterFile, masterField, "InitVal_recon", temp, resolveFlag);
	reconParams->InitVal_recon = str2double(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "initReconMode", temp, resolveFlag);
	strcpy(reconParams->initReconMode, temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "priorWeight_QGGMRF", temp, resolveFlag);
	reconParams->priorWeight_QGGMRF = str2double(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "priorWeight_proxMap", temp, resolveFlag);
	reconParams->priorWeight_proxMap = str2double(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "q", temp, resolveFlag);
	reconParams->q = str2double(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "p", temp, resolveFlag);
	reconParams->p = str2double(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "T", temp, resolveFlag);
	reconParams->T = str2double(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "sigmaX", temp, resolveFlag);
	reconParams->sigmaX = str2double(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "bFace", temp, resolveFlag);
	reconParams->bFace = str2double(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "bEdge", temp, resolveFlag);
	reconParams->bEdge = str2double(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "bVertex", temp, resolveFlag);
	reconParams->bVertex = str2double(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "sigma_lambda", temp, resolveFlag);
	reconParams->sigma_lambda = str2double(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "is_positivity_constraint", temp, resolveFlag);
	reconParams->is_positivity_constraint = str2int(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "isTGGMRF", temp, resolveFlag);
	reconParams->isTGGMRF = str2int(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "stopThresholdChange_pct", temp, resolveFlag);
	reconParams->stopThresholdChange_pct = str2double(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "stopThesholdRWFE_pct", temp, resolveFlag);
	reconParams->stopThesholdRWFE_pct = str2double(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "stopThesholdRUFE_pct", temp, resolveFlag);
	reconParams->stopThesholdRUFE_pct = str2double(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "MaxIterations", temp, resolveFlag);
	reconParams->MaxIterations = str2int(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "relativeChangeMode", temp, resolveFlag);
	strcpy(reconParams->relativeChangeMode, temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "relativeChangeScaler", temp, resolveFlag);
	reconParams->relativeChangeScaler = str2double(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "relativeChangePercentile", temp, resolveFlag);
	reconParams->relativeChangePercentile = str2double(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "downsampleFactorSino", temp, resolveFlag);
	reconParams->downsampleFactorSino = str2int(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "downsampleFactorRecon", temp, resolveFlag);
	reconParams->downsampleFactorRecon = str2int(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "downsampleFNamePrefix", temp, resolveFlag);
	strcpy(reconParams->downsampleFNamePrefix, temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "N_G", temp, resolveFlag);
	reconParams->N_G = str2int(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "zipLineMode", temp, resolveFlag);
	reconParams->zipLineMode = str2int(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "numVoxelsPerZiplineMax", temp, resolveFlag);
	reconParams->numVoxelsPerZiplineMax = str2int(temp);

	reconParams->numVoxelsPerZipline = -1;
	reconParams->numZiplines = -1;

	plainParams(plainParamsFile, get_set, masterFile, masterField, "numThreads", temp, resolveFlag);
	reconParams->numThreads = str2int(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "weightScaler_estimateMode", temp, resolveFlag);
	strcpy(reconParams->weightScaler_estimateMode, temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "weightScaler_domain", temp, resolveFlag);
	strcpy(reconParams->weightScaler_domain, temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "weightScaler_value", temp, resolveFlag);
	reconParams->weightScaler_value = str2double(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "NHICD_Mode", temp, resolveFlag);
	strcpy(reconParams->NHICD_Mode, temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "NHICD_ThresholdAllVoxels_ErrorPercent", temp, resolveFlag);
	reconParams->NHICD_ThresholdAllVoxels_ErrorPercent = str2double(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "NHICD_percentage", temp, resolveFlag);
	reconParams->NHICD_percentage = str2double(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "NHICD_random", temp, resolveFlag);
	reconParams->NHICD_random = str2double(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "verbosity", temp, resolveFlag);
	reconParams->verbosity = str2int(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "isComputeCost", temp, resolveFlag);
	reconParams->isComputeCost = str2int(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "backprojlike_type", temp, resolveFlag);
	strcpy(reconParams->backprojlike_type, temp);


	/* if updated, also update "printReconParams" */

}

void readAndAllocateViewAngleList(char *masterFile, char *plainParamsFile, struct ViewAngleList *list, struct SinoParams *sinoParams)
{
   	FILE *fid;
   	long int i, N_beta;
	char get_set[1000] = "get";
	char masterField[1000] = "viewAngleList";
	char fName_rel[1000] = "";
	char fName[1000] = "";
	char resolveFlag[100] = "-r";

	plainParams(plainParamsFile, get_set, masterFile, masterField, "", fName_rel, resolveFlag);
	absolutePath_relativeToFileLocation(fName_rel, masterFile, fName);


	fid = fopen(fName, "r");
    if (fid == NULL)
    {
        fprintf(stderr, "ERROR in readViewAngleList: Failed to open file \"%s\"", fName);
        exit(-1);
    }
    list->N_beta = 0;
    fscanf(fid, "%ld", &list->N_beta);
   	
   	if(list->N_beta != sinoParams->N_beta)
   	{
        fprintf(stderr, "ERROR in readViewAngleList: Number of view angles does not match");
        fprintf(stderr, "\t(Angle list) N_beta = %ld\n", list->N_beta);
        fprintf(stderr, "\t(sinoParams) N_beta = %ld\n", sinoParams->N_beta);
        exit(-1);
   	}
   	N_beta = list->N_beta;

   	list->beta = (double*) mem_alloc_1D(N_beta, sizeof(double));

   	for (i = 0; i < N_beta; ++i)
   	{
   		fscanf(fid, "%lf", &list->beta[i]);
   	}

    fclose(fid);
}


void printPathNames(struct PathNames *pathNames)
{
	printf("\nPath Names Read:\n");

	printf("\tmasterFile = %s \n", pathNames->masterFile);
	printf("\tplainParamsFile = %s \n", pathNames->plainParamsFile);
	printf("\tsino = %s \n", pathNames->sino);
	printf("\tdriftSino = %s \n", pathNames->driftSino);
	printf("\twght = %s \n", pathNames->wght);
	printf("\terrSino = %s \n", pathNames->errSino);
	printf("\trecon = %s \n", pathNames->recon);
	printf("\tproxMapInput = %s \n", pathNames->proxMapInput);
	printf("\tlastChange = %s \n", pathNames->lastChange);
	printf("\ttimeToChange = %s \n", pathNames->timeToChange);
	printf("\tsysMatrix = %s \n", pathNames->sysMatrix);
	printf("\twghtRecon = %s \n", pathNames->wghtRecon);
	printf("\tprojInput = %s \n", pathNames->projInput);
	printf("\tprojOutput = %s \n", pathNames->projOutput);
	printf("\tbackprojlikeInput = %s \n", pathNames->backprojlikeInput);
	printf("\tbackprojlikeOutput = %s \n", pathNames->backprojlikeOutput);
	printf("\testimateSino = %s \n", pathNames->estimateSino);
	printf("\tconsensusRecon = %s \n", pathNames->consensusRecon);
	printf("\tjigMeasurementsSino = %s \n", pathNames->jigMeasurementsSino);

}


int getDataTypeSize(char *dataType)
{
	if (strcmp(dataType, "char") == 0) 
	{
		return sizeof(char);
	} 
	else if (strcmp(dataType, "float") == 0)
	{
		return sizeof(float);
	}
	else if (strcmp(dataType, "long int") == 0)
	{
		return sizeof(long int);
	}
	else if (strcmp(dataType, "unsigned char") == 0)
	{
		return sizeof(unsigned char);
	}	else /* default: */
	{
		printf("ERROR: Error in getDataTypeSize: data type \"%s\" not recognized\n", dataType);
		exit(-1);
	}
}

void prependToFName(char *prefix, char *fName)
{
	char bName[1000], dName[1000], fName_tmp[1000];

	strcpy(fName_tmp, fName);
	strcpy(dName, dirname(fName_tmp));

	strcpy(fName_tmp, fName);
	strcpy(bName, basename(fName_tmp));

	sprintf(fName, "%s/%s%s", dName, prefix, bName);
}

void read3DData(char *fName, void ***arr, long int N1, long int N2, long int N3, char *dataType)
{
	FILE* fp;
	int buff[3];
	long int N1_is, N2_is, N3_is;
	long int numToRead;
	long int numHaveRead;

	int dataTypeSize;

	dataTypeSize = getDataTypeSize(dataType);

	fp = fopen (fName, "r" );
	if(fp == NULL)
	{
		printf("Error in opening file \"%s\".\n", fName);
		goto error;
	}

	/**
	 * 		Read dimesions in header
	 */
	numToRead = 3;
	numHaveRead = fread(&buff, sizeof(int), numToRead, fp);
	if (numHaveRead != numToRead)
	{
		printf("Error in reading header in file \"%s\".\n", fName);
		printf("Number of elements read does not match required\n");
		printf("numToRead = %ld, numHaveRead = %ld\n", numToRead, numHaveRead);
		goto error;
	}

	/**
	 * 		Check if header dimensions match the input dimensions
	 */
	N1_is = buff[0];
	N2_is = buff[1];
	N3_is = buff[2];
	if (N1_is != N1 || N2_is != N2 || N3_is != N3)
	{
		printf("Error in reading file \"%s\".\n", fName);
		printf("Dimensions don't match\n");
		printf("Dimensions should: (%3ld, %3ld, %3ld)\n", N1, N2, N3);
		printf("Dimensions   is  : (%3ld, %3ld, %3ld)\n", N1_is, N2_is, N3_is);
		goto error;
	}

	/**
	 * 		Read body of data
	 */
	numToRead = N1*N2*N3;
	numHaveRead = fread(**arr, dataTypeSize, numToRead, fp);
	if (numHaveRead != numToRead)
	{
		printf("Error in reading body file \"%s\".\n", fName);
		printf("Number of elements read does not match required\n");
		printf("numToRead = %ld, numHaveRead = %ld\n", numToRead, numHaveRead);
		goto error;
	}

	fclose(fp);
	return;


error:
	if (fp)
		fclose(fp);
	exit(-1);
}

void write3DData(char *fName, void ***arr, long int N1, long int N2, long int N3, char *dataType)
{
	FILE* fp;
	int buff[3];
	long int numToWrite;
	long int numWritten;

	int dataTypeSize;

	dataTypeSize = getDataTypeSize(dataType);

	fp = fopen (fName, "w" );
	if(fp == NULL)
	{
		printf("Error in opening file \"%s\".\n", fName);
		goto error;
	}

	/**
	 * 		Write dimesions in header
	 */
	buff[0] = N1;
	buff[1] = N2;
	buff[2] = N3;

	numToWrite = 3;
	numWritten = fwrite(&buff, sizeof(int), numToWrite, fp);
	if (numWritten != numToWrite)
	{
		printf("Error in writing header in file \"%s\".\n", fName);
		printf("Number of elements written does not match required\n");
		printf("numToWrite = %ld, numWritten = %ld\n", numToWrite, numWritten);
		goto error;
	}

	/**
	 * 		Write body of data
	 */
	numToWrite = N1*N2*N3;
	numWritten = fwrite(**arr, dataTypeSize, numToWrite, fp);
	if (numWritten != numToWrite)
	{
		printf("Error in reading body file \"%s\".\n", fName);
		printf("Number of elements written does not match required\n");
		printf("numToWrite = %ld, numWritten = %ld\n", numToWrite, numWritten);
		goto error;
	}

	fclose(fp);
	return;


error:
	if (fp)
		fclose(fp);
	exit(-1);
}



void writeSinoData3DCone(char *fName, void ***sino, struct SinoParams *sinoParams, char *dataType)
{

    write3DData(fName, (void***)sino, sinoParams->N_beta, sinoParams->N_dv, sinoParams->N_dw, dataType);
}

void readSinoData3DCone(char *fName, void ***sino, struct SinoParams *sinoParams, char *dataType)
{
    read3DData(fName, (void***)sino, sinoParams->N_beta, sinoParams->N_dv, sinoParams->N_dw, dataType);
}

void writeImageData3DCone(char *fName, void ***arr, struct ImageParams *params, int isROI, char *dataType)
{
    if (isROI) 
    {
        write3DData(fName, (void***)arr, params->N_x_roi, params->N_y_roi, params->N_z_roi, dataType);
    }
    else
    {
        write3DData(fName, (void***)arr, params->N_x, params->N_y, params->N_z, dataType);
    }
}

void readImageData3DCone(char *fName, void ***arr, struct ImageParams *params, int isROI, char *dataType)
{
    if (isROI) 
    {
        read3DData(fName, (void***)arr, params->N_x_roi, params->N_y_roi, params->N_z_roi, dataType);
    }
    else
    {
        read3DData(fName, (void***)arr, params->N_x, params->N_y, params->N_z, dataType);
    }
}





