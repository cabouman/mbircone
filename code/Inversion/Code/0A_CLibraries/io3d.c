
#include "io3d.h"
#include <libgen.h>
#include "../../../plainParams/plainParams.h"



/* Read Command-line */
void readCmdLine(int argc, char *argv[], struct CmdLine *cmdLine)
{
	char ch;

	

	cmdLine->masterFile[0] = '\0';
	cmdLine->masterFile[0] = '\0';

	/* get options */
	while ((ch = getopt(argc, argv, "a:b:")) != EOF)
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
}

void printCmdLine(struct CmdLine *cmdLine)
{
	char str[500];
	sprintf(str, 	"\n"
					"Command line arguments read:\n"
					"\tmasterFile file = %s\n"
					"\tplainParamsFile file = %s\n", cmdLine->masterFile, cmdLine->plainParamsFile);					
	logAndDisp_message(LOG_PROGRESS, str);
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

	plainParams(plainParamsFile, get_set, masterFile, masterField, "errsino", pathNames->errsino, resolveFlag);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "sinoMask", pathNames->sinoMask, resolveFlag);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "recon", pathNames->recon, resolveFlag);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "reconROI", pathNames->reconROI, resolveFlag);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "proxMapInput", pathNames->proxMapInput, resolveFlag);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "lastChange", pathNames->lastChange, resolveFlag);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "timeToChange", pathNames->timeToChange, resolveFlag);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "phantom", pathNames->phantom, resolveFlag);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "sysMatrix", pathNames->sysMatrix, resolveFlag);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "wghtRecon", pathNames->wghtRecon, resolveFlag);


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
	sinoParams->weightScaler = -1.0;

}

void readImageFParams(char *masterFile, char *plainParamsFile, struct ImageFParams *imgParams)
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

	plainParams(plainParamsFile, get_set, masterFile, masterField, "isUsePhantomToInitErrSino", temp, resolveFlag);
	reconParams->isUsePhantomToInitErrSino = str2int(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "InitVal_proxMapInput", temp, resolveFlag);
	reconParams->InitVal_proxMapInput = str2double(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "rho", temp, resolveFlag);
	reconParams->rho = str2double(temp);

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

	plainParams(plainParamsFile, get_set, masterFile, masterField, "stopThesholdRRMSE_pct", temp, resolveFlag);
	reconParams->stopThesholdRRMSE_pct = str2double(temp);

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

	plainParams(plainParamsFile, get_set, masterFile, masterField, "isEstimateWeightScaler", temp, resolveFlag);
	reconParams->isEstimateWeightScaler = str2int(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "isUseWghtRecon", temp, resolveFlag);
	reconParams->isUseWghtRecon = str2int(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "weightScaler", temp, resolveFlag);
	reconParams->weightScaler = str2double(temp);

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

	plainParams(plainParamsFile, get_set, masterFile, masterField, "isPhantomPresent", temp, resolveFlag);
	reconParams->isPhantomPresent = str2int(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "isForwardProjectPhantom", temp, resolveFlag);
	reconParams->isForwardProjectPhantom = str2int(temp);

	plainParams(plainParamsFile, get_set, masterFile, masterField, "isRecomputeWeight", temp, resolveFlag);
	reconParams->isRecomputeWeight = str2int(temp);

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
	char str[10000];

	sprintf(str, "\nPath Names Read:\n");

	sprintf(str, "%s\tmasterFile = %s \n", str, pathNames->masterFile);
	sprintf(str, "%s\tplainParamsFile = %s \n", str, pathNames->plainParamsFile);
	sprintf(str, "%s\n", str);
	sprintf(str, "%s\tsino = %s \n", str, pathNames->sino);
	sprintf(str, "%s\tdriftSino = %s \n", str, pathNames->driftSino);
	sprintf(str, "%s\twght = %s \n", str, pathNames->wght);
	sprintf(str, "%s\terrsino = %s \n", str, pathNames->errsino);
	sprintf(str, "%s\tsinoMask = %s \n", str, pathNames->sinoMask);
	sprintf(str, "%s\trecon = %s \n", str, pathNames->recon);
	sprintf(str, "%s\treconROI = %s \n", str, pathNames->reconROI);
	sprintf(str, "%s\tproxMapInput = %s \n", str, pathNames->proxMapInput);
	sprintf(str, "%s\tlastChange = %s \n", str, pathNames->lastChange);
	sprintf(str, "%s\ttimeToChange = %s \n", str, pathNames->timeToChange);
	sprintf(str, "%s\tphantom = %s \n", str, pathNames->phantom);
	sprintf(str, "%s\tsysMatrix = %s \n", str, pathNames->sysMatrix);
	sprintf(str, "%s\twghtRecon = %s \n", str, pathNames->wghtRecon);

	logAndDisp_message(LOG_PROGRESS, str);
}


void printSinoParams(struct SinoParams *params)
{
	char str[2000];

	sprintf(str, "\nSinogram parameters read:\n");

	sprintf(str, "%s\tN_dv = %ld,\n", str, params->N_dv);
	sprintf(str, "%s\tN_dw = %ld,\n", str, params->N_dw);
	sprintf(str, "%s\tDelta_dv = %e,\n", str, params->Delta_dv);
	sprintf(str, "%s\tDelta_dw = %e,\n", str, params->Delta_dw);
	sprintf(str, "%s\tN_beta = %ld,\n", str, params->N_beta);
	sprintf(str, "%s\tu_s = %e,\n", str, params->u_s);
	sprintf(str, "%s\tu_r = %e,\n", str, params->u_r);
	sprintf(str, "%s\tv_r = %e,\n", str, params->v_r);
	sprintf(str, "%s\tu_d0 = %e,\n", str, params->u_d0);
	sprintf(str, "%s\tv_d0 = %e,\n", str, params->v_d0);
	sprintf(str, "%s\tw_d0 = %e,\n", str, params->w_d0);
	sprintf(str, "%s\t(potentially uninitialized:)\n", str);
	sprintf(str, "%s\tweightScaler = %e,\n", str, params->weightScaler);
	sprintf(str, "%s\n", str);
	logAndDisp_message(LOG_PROGRESS, str);
}

void printImgParams(struct ImageFParams *params)
{
	char str[2000];
	sprintf(str, "\nImageF parameters read:\n");

	sprintf(str, "%s\tx_0 = %e \n", str, params->x_0);
	sprintf(str, "%s\ty_0 = %e \n", str, params->y_0);
	sprintf(str, "%s\tz_0 = %e \n", str, params->z_0);
	sprintf(str, "%s\tN_x = %ld \n", str, params->N_x);
	sprintf(str, "%s\tN_y = %ld \n", str, params->N_y);
	sprintf(str, "%s\tN_z = %ld \n", str, params->N_z);
	sprintf(str, "%s\tDelta_xy = %e \n", str, params->Delta_xy);
	sprintf(str, "%s\tDelta_z = %e \n", str, params->Delta_z);
	sprintf(str, "%s\tj_xstart_roi = %ld \n", str, params->j_xstart_roi);
	sprintf(str, "%s\tj_ystart_roi = %ld \n", str, params->j_ystart_roi);
	sprintf(str, "%s\tj_zstart_roi = %ld \n", str, params->j_zstart_roi);
	sprintf(str, "%s\tj_xstop_roi = %ld \n", str, params->j_xstop_roi);
	sprintf(str, "%s\tj_ystop_roi = %ld \n", str, params->j_ystop_roi);
	sprintf(str, "%s\tj_zstop_roi = %ld \n", str, params->j_zstop_roi);
	sprintf(str, "%s\n", str);

	logAndDisp_message(LOG_PROGRESS, str);

}


void printReconParams(struct ReconParams *params)
{
	char str[2000];
	str[0] = '\0';

	sprintf(str, "%s\nReconstruction parameters read:\n", str);
	
	sprintf(str, "%s\tInitVal_recon = %e \n", str, params->InitVal_recon);
	sprintf(str, "%s\tisUsePhantomToInitErrSino = %d \n", str, params->isUsePhantomToInitErrSino);
	sprintf(str, "%s\tInitVal_proxMapInput = %e \n", str, params->InitVal_proxMapInput);
	sprintf(str, "%s\trho = %e \n", str, params->rho);
	sprintf(str, "%s\tpriorWeight_QGGMRF = %e \n", str, params->priorWeight_QGGMRF);
	sprintf(str, "%s\tpriorWeight_proxMap = %e \n", str, params->priorWeight_proxMap);
	sprintf(str, "%s\tq = %e \n", str, params->q);
	sprintf(str, "%s\tp = %e \n", str, params->p);
	sprintf(str, "%s\tT = %e \n", str, params->T);
	sprintf(str, "%s\tsigmaX = %e \n", str, params->sigmaX);
	sprintf(str, "%s\tbFace = %e \n", str, params->bFace);
	sprintf(str, "%s\tbEdge = %e \n", str, params->bEdge);
	sprintf(str, "%s\tbVertex = %e \n", str, params->bVertex);
	sprintf(str, "%s\tsigma_lambda = %e \n", str, params->sigma_lambda);
	sprintf(str, "%s\tis_positivity_constraint = %d \n", str, params->is_positivity_constraint);
	sprintf(str, "%s\tisTGGMRF = %d \n", str, params->isTGGMRF);

	
	sprintf(str, "%s\tstopThresholdChange_pct = %e \n", str, params->stopThresholdChange_pct);
	sprintf(str, "%s\tstopThesholdRWFE_pct = %e \n", str, params->stopThesholdRWFE_pct);
	sprintf(str, "%s\tstopThesholdRUFE_pct = %e \n", str, params->stopThesholdRUFE_pct);
	sprintf(str, "%s\tstopThesholdRRMSE_pct = %e \n", str, params->stopThesholdRRMSE_pct);
	sprintf(str, "%s\tMaxIterations = %d \n", str, params->MaxIterations);
	sprintf(str, "%s\trelativeChangeMode = %s \n", str, params->relativeChangeMode);
	sprintf(str, "%s\trelativeChangeScaler = %e \n", str, params->relativeChangeScaler);
	sprintf(str, "%s\trelativeChangePercentile = %e \n", str, params->relativeChangePercentile);

	sprintf(str, "%s\tdownsampleFactorSino = %d \n", str, params->downsampleFactorSino);
	sprintf(str, "%s\tdownsampleFactorRecon = %d \n", str, params->downsampleFactorRecon);
	sprintf(str, "%s\tdownsampleFNamePrefix = %s \n", str, params->downsampleFNamePrefix);
	sprintf(str, "%s\tN_G = %d \n", str, params->N_G);
	sprintf(str, "%s\tzipLineMode = %d \n", str, params->zipLineMode);
	sprintf(str, "%s\tnumVoxelsPerZiplineMax = %d \n", str, params->numVoxelsPerZiplineMax);
	sprintf(str, "%s\tnumVoxelsPerZipline = %d \n", str, params->numVoxelsPerZipline);
	sprintf(str, "%s\tnumZiplines = %d \n", str, params->numZiplines);
	sprintf(str, "%s\tnumThreads = %d \n", str, params->numThreads);
	sprintf(str, "%s\tisEstimateWeightScaler = %d \n", str, params->isEstimateWeightScaler);
	sprintf(str, "%s\tisUseWghtRecon = %d \n", str, params->isUseWghtRecon);
	sprintf(str, "%s\tweightScaler = %e \n", str, params->weightScaler);

	sprintf(str, "%s\tNHICD_Mode = %s \n", str, params->NHICD_Mode);
	sprintf(str, "%s\tNHICD_ThresholdAllVoxels_ErrorPercent = %e \n", str, params->NHICD_ThresholdAllVoxels_ErrorPercent);
	sprintf(str, "%s\tNHICD_percentage = %e \n", str, params->NHICD_percentage);
	sprintf(str, "%s\tNHICD_random = %e \n", str, params->NHICD_random);

	sprintf(str, "%s\tverbosity = %d \n", str, params->verbosity);
	sprintf(str, "%s\tisComputeCost = %d \n", str, params->isComputeCost);
	sprintf(str, "%s\tisPhantomPresent = %d \n", str, params->isPhantomPresent);
	sprintf(str, "%s\tisForwardProjectPhantom = %d \n", str, params->isForwardProjectPhantom);
	sprintf(str, "%s\tisRecomputeWeight = %d \n", str, params->isRecomputeWeight);

	sprintf(str, "%s\n", str);

	logAndDisp_message(LOG_PROGRESS, str);
}

void printSysMatrixParams(struct SysMatrix *A)
{
	char str[1000];

	sprintf(str, "\nSystemMatrix parameters:\n");

	sprintf(str, "%s\ti_vstride_max = %ld \n", str, A->i_vstride_max);
	sprintf(str, "%s\ti_wstride_max = %ld \n", str, A->i_wstride_max);
	sprintf(str, "%s\tN_u = %ld \n", str, A->N_u);
	sprintf(str, "%s\tDelta_u = %e \n", str, A->Delta_u);
	sprintf(str, "%s\tu_0 = %e \n", str, A->u_0);
	sprintf(str, "%s\tu_1 = %e \n", str, A->u_1);
	sprintf(str, "%s\tB_ij_max = %e \n", str, A->B_ij_max);
	sprintf(str, "%s\tC_ij_max = %e \n", str, A->C_ij_max);
	sprintf(str, "%s\tB_ij_scaler = %e \n", str, A->B_ij_scaler);
	sprintf(str, "%s\tC_ij_scaler = %e \n", str, A->C_ij_scaler);
	sprintf(str, "%s\n", str);

	logAndDisp_message(LOG_PROGRESS, str);
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
	numHaveRead = fread(&arr[0][0][0], dataTypeSize, numToRead, fp);
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
	numWritten = fwrite(&arr[0][0][0], dataTypeSize, numToWrite, fp);
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


void writeDownSampledFloat3D(char *fName, float ***arr, long int N1, long int N2, long int N3, int D1, int D2, int D3)
{
	float ***arr_DS;
	int N1DS, N2DS, N3DS;
	int i1, i2, i3;

	D1 = _MIN_(D1, N1-1);
	D2 = _MIN_(D2, N2-1);
	D3 = _MIN_(D3, N3-1);

	N1DS = N1/D1;
	N2DS = N2/D2;
	N3DS = N3/D3;


	arr_DS = (float***) mem_alloc_3D( N1DS,  N2DS,  N3DS, sizeof(float));


	for (i1 = 0; i1 < N1DS; ++i1)
	for (i2 = 0; i2 < N2DS; ++i2)
	for (i3 = 0; i3 < N3DS; ++i3)
	{
		arr_DS[i1][i2][i3] = arr[i1*D1][i2*D2][i3*D3];
	}

	write3DData(fName, (void***)arr_DS, N1DS, N2DS, N3DS, "float");

	mem_free_3D((void***)arr_DS);


}



long int keepWritingToBinaryFile(FILE *fp, void *var, long int numEls, int elSize, char *fName)
{
	/* Return number of bytes written */
	long int numElsWritten;

    numElsWritten = fwrite(var, elSize, numEls, fp);
    if(numElsWritten != numEls)
    {
        fprintf(stderr, "ERROR in keepWritingToBinaryFile: file \"%s\" terminated early.\n", fName);
        fprintf(stderr, "Tried to write %li elements of size %d Bytes. Wrote %li elements.\n", numEls, elSize, numElsWritten);

        fclose(fp);
        exit(-1);
    }
    return (long int) numEls * elSize;
}

long int keepReadingFromBinaryFile(FILE *fp, void *var, long int numEls, int elSize, char *fName)
{
	/* Return number of bytes read */
	long int numElsRead;

    numElsRead = fread(var, elSize, numEls, fp);
    if(numElsRead != numEls)
    {
        fprintf(stderr, "ERROR in keepReadingFromBinaryFile: file \"%s\" terminated early.\n", fName);
        fprintf(stderr, "Tried to read %li elements of size %d Bytes. Read %li elements.\n", numEls, elSize, numElsRead);
        fclose(fp);
        exit(-1);
    }
    return (long int) numEls * elSize;
}


void printFileIOInfo( char* functionName, char* fName, long int size, char mode)
{
	char readwrite[200];	/* puts the word "Read" or "Write" into the output */
	switch(mode)
	{
		case 'r': 	strcpy(readwrite, "Read "); break;
		case 'w': 	strcpy(readwrite, "Write"); break;
		default:	printf("Error in printFileIOInfo: Use mode 'r' or 'w'\n");
					exit(-1);
	}
	printf("\n");
	printf("    ************** FILE ACCESS ********************************\n");
	printf(" ****  File access in: %s\n", functionName);
	printf("*****  File name     : %s\n", fName);
	printf("*****  %-14s: %-15ld bytes\n", readwrite, size);
	printf("*****                = %-15e kB\n", (double) size*1e-3);
	printf(" ****                = %-15e MB\n", (double) size*1e-6);
	printf("    ***********************************************************\n");
}

void printProgressOfLoop( long int indexOfLoop, long int NumIterations)
{
	double percent;

	percent = (double) (1+indexOfLoop) / (double) NumIterations * 100;
	printf("\r[%.1e%%]", percent );
	fflush(stdout);	

}

void logAndDisp_message(char *fName, char* message)
{
	log_message(fName, message);
	printf("%s", message);
}

void log_message(char *fName, char* message)
{
	FILE *fp;

	fp = fopen(fName, "a");
	if (fp != NULL)
	{
		fprintf(fp, "%s", message);
		fclose(fp);
	}
	else
	{
		fprintf(stderr, "WARNING: In log_message: Could not open file %s\n", fName);
	}

}


void resetFile(char *fName)
{
	FILE *filePointer;

	filePointer = fopen(fName, "w");
	fclose(filePointer);
		
}







