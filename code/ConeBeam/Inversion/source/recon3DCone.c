
#include "recon3DCone.h"
#include "allocate.h"
#include <math.h>
#include <time.h>
#include <omp.h>




void MBIR3DCone(struct ImageF *img, struct Sino *sino, struct ReconParams *reconParams, struct SysMatrix *A, struct PathNames *pathNames)
{
	int itNumber = 0, MaxIterations;
    double stopThresholdChange;
    double stopThesholdRWFE, stopThesholdRUFE;
    double stopThesholdRRMSE;
    double RRMSE = -1.0;
	long int j_xy, j_x, j_y, j_z;
	long int j_xyz;
	long int N_x, N_y, N_z;
	long int N_beta, N_dv, N_dw;
	long int k_G, N_G;
	long int numZiplines;
	long int numVoxelsInMask;
    double ratioUpdated;
    double relUpdate;
	char tempFName[1000];

	double timer_icd_loop;
	double ticToc_icdUpdate;
	double ticToc_icdUpdate_total;
	double ticToc_all;
	double ticToc_randomization;
	double ticToc_computeCost;
	double ticToc_computeRelUpdate;
	double ticToc_IntermediateStoring;
	double ticToc_iteration;
	double ticToc_computeLastChangeThreshold;

	char stopFlag = 0;

	struct ICDInfo3DCone *icdInfoArray;			/* Only used when using zip line option*/
	struct ICDInfo3DCone icdInfo;				/* Only used when not using zip line option*/
	struct ParallelAux parallelAux;

	/* Hardcoded stuff */
	int subsampleFactor = 10;


	/* Iteration statistics */
	double cost = -1.0;
	double weightedNormSquared_e, weightedNormSquared_y;
	double normSquared_e, normSquared_y;


	struct SpeedAuxICD speedAuxICD;
    struct ReconAux reconAux;

	/* Renaming some variables */
	MaxIterations = reconParams->MaxIterations;
    stopThresholdChange = reconParams->stopThresholdChange_pct/100.0;
    stopThesholdRWFE = reconParams->stopThesholdRWFE_pct/100.0;
    stopThesholdRUFE = reconParams->stopThesholdRUFE_pct/100.0;
    stopThesholdRRMSE = reconParams->stopThesholdRRMSE_pct/100.0;
	N_x = img->params.N_x;
	N_y = img->params.N_y;
	N_z = img->params.N_z;
	N_beta = sino->params.N_beta;
	N_dv = sino->params.N_dv;
	N_dw = sino->params.N_dw;
	numZiplines = reconParams->numZiplines;

    /* reconAux*/
    reconAux.NHICD_isPartialUpdateActive = 0;
	reconAux.lastChangeThreshold = -1;
    reconAux.N_M_max = ceil(img->params.N_z/2.0);
    reconAux.totalEquits = 0;
    reconAux.TotalValueChange = 0.0;
	reconAux.TotalVoxelValue = 0.0;
	reconAux.NumUpdatedVoxels = 0.0;
	reconAux.NHICD_neighborFilter[0][0] = reconAux.NHICD_neighborFilter[0][2] = reconAux.NHICD_neighborFilter[2][0] = reconAux.NHICD_neighborFilter[2][2] = 0.1036;
	reconAux.NHICD_neighborFilter[1][0] = reconAux.NHICD_neighborFilter[0][1] = reconAux.NHICD_neighborFilter[2][1] = reconAux.NHICD_neighborFilter[1][2] = 0.1464;
	reconAux.NHICD_neighborFilter[1][1] = 0.0;
	
	reconAux.relativeWeightedForwardError = 0;
	reconAux.relativeUnweightedForwardError = 0;
	reconAux.NHICD_numUpdatedVoxels = (long int*) malloc(numZiplines*sizeof(long int));
	reconAux.NHICD_totalValueChange = (double*) malloc(numZiplines*sizeof(double));
	reconAux.NHICD_isPartialZiplineHot = (int*) malloc(numZiplines*sizeof(int));



	srand(0);

    numVoxelsInMask = computeNumVoxelsInImageMask(img);


    /**
     * 		Random Auxiliary
     */
	RandomAux_allocate(&img->randomAux, &img->params);
	RandomAux_Initialize(&img->randomAux, &img->params);

	RandomZiplineAux_allocate(&img->randomZiplineAux, &img->params, reconParams);
	RandomZiplineAux_Initialize(&img->randomZiplineAux, &img->params, reconParams, reconAux.N_M_max);

	N_G = img->randomZiplineAux.N_G;

	/**
	 * 		Parallel stuff
	 */
	/*printReconParams(reconParams);*/
	omp_set_num_threads(reconParams->numThreads);
	prepareParallelAux(&parallelAux, reconAux.N_M_max);

	/**
	 * 		Super Voxel Stuff   
	 */



	/**
	 * 		Print stuff
	 */
/*	printPathNames(pathNames);
	printReconParams(reconParams);
	printSinoParams(&sino->params);
	printImgParams(&img->params);*/

    /**
     * 		Loop initialization
     */
	icdInfoArray = mem_alloc_1D(reconAux.N_M_max, sizeof(struct ICDInfo3DCone));
	resetFile(LOG_ICDLOOP);
	resetFile(LOG_STATS);

	timer_reset(&timer_icd_loop);
	tic(&ticToc_all);
	ticToc_icdUpdate_total = 0;

	for (itNumber = 0; (itNumber <= MaxIterations) && (stopFlag==0); ++itNumber) 
	{

		/**
		 * 		Shuffle the order in which the voxels are updated
		 * 		before every iteration
		 */
		tic(&ticToc_randomization);
		tic(&ticToc_iteration);
		switch(reconParams->zipLineMode)
		{
			case 0: /* off */
			RandomAux_ShuffleOrderXYZ(&img->randomAux, &img->params);
			break;
			case 1: /* conventional zipline */
			RandomZiplineAux_ShuffleGroupIndices_FixedDistance(&img->randomZiplineAux, &img->params);
			break;
			case 2: /* randomized zipline */
			RandomZiplineAux_ShuffleGroupIndices(&img->randomZiplineAux, &img->params);
			break;
			default:
			printf("Error: zipLineMode unknown\n");
			exit(-1);
		}
		toc(&ticToc_randomization);


		tic(&ticToc_icdUpdate);
		speedAuxICD_reset(&speedAuxICD);
		resetIterationStats(&reconAux);
		if (itNumber>0)
		{

			/*###############################################################################*/
			/**
			 * 		Update all voxels
			 */
			/*###############################################################################*/
			if(reconParams->zipLineMode == 1 || reconParams->zipLineMode == 2)
			{
				/********************************************************************************************/
				/**
				 * 		ICD Zipline
				 */
				/********************************************************************************************/
					RandomZiplineAux_shuffleOrderXY(&img->randomZiplineAux, &img->params);

					for (j_xy = 0; j_xy < N_x*N_y; ++j_xy)
					{

						if (timer_hasPassed(&timer_icd_loop, OUTPUT_REFRESH_TIME))
						{
							speedAuxICD_computeSpeed(&speedAuxICD);
							writeICDLoopStatus2File(LOG_ICDLOOP, j_xy, N_x*N_y, itNumber, speedAuxICD.voxelsPerSecond);
						}
						

						/**
						 * 		Prepare icdInfo for whole zip line
						 */
						indexExtraction2D(img->randomZiplineAux.orderXY[j_xy], &j_x, N_x, &j_y, N_y);
						if (isInsideMask(j_x, j_y, N_x, N_y))
						{

							/*prepareNHICDStats(&reconAux);*/
							NHICD_checkPartialZiplinesHot(&reconAux, j_x, j_y, reconParams, img);


							for (k_G = 0; k_G < N_G; ++k_G)
							{
								img->randomZiplineAux.k_G = k_G;
								prepareICDInfoRandGroup(j_x, j_y, &img->randomZiplineAux, icdInfoArray, img, reconParams, &reconAux);

								ICDStep3DConeGroup(sino, img, A, icdInfoArray, reconParams, &img->randomZiplineAux, &parallelAux, &reconAux);

								updateIterationStatsGroup(&reconAux, icdInfoArray, &img->randomZiplineAux, img, reconParams);
							}
							speedAuxICD_update(&speedAuxICD, img->randomZiplineAux.N_M);
							updateNHICDStats(&reconAux, j_x, j_y, img, reconParams);
						}
					}
			}


			if(reconParams->zipLineMode == 0)
			{
				/********************************************************************************************/
				/**
				 * 		ICD Single voxel updates
				 */
				/********************************************************************************************/
				for (j_xyz = 0; j_xyz < N_x*N_y*N_z; ++j_xyz)
				{
					if (timer_hasPassed(&timer_icd_loop, OUTPUT_REFRESH_TIME))
					{
						speedAuxICD_computeSpeed(&speedAuxICD);
						writeICDLoopStatus2File(LOG_ICDLOOP, j_xyz, N_x*N_y*N_z, itNumber, speedAuxICD.voxelsPerSecond);
					}

					/**
					 * 		Prepare icdInfo
					 */
					indexExtraction3D(img->randomAux.orderXYZ[j_xyz], &j_x, N_x, &j_y, N_y, &j_z, N_z);
					if (isInsideMask(j_x, j_y, N_x, N_y) && (!reconAux.NHICD_isPartialUpdateActive || NHICD_isVoxelHot(reconParams, img, j_x, j_y, j_z, reconAux.lastChangeThreshold)))
					{
						prepareICDInfo(j_x, j_y, j_z, &icdInfo, img, &reconAux, reconParams);

						/**
						 * 		ICD update of one voxel
						 */
						ICDStep3DCone(sino, img, A, &icdInfo, reconParams, &reconAux);

						/**
						 * 		Update iteration statistics
						 */
						updateIterationStats(&reconAux, &icdInfo, img);
						speedAuxICD_update(&speedAuxICD, 1);
					}
				}
			}
		}
		printf("\r                                                                                    \r");
		speedAuxICD_computeSpeed(&speedAuxICD);
		toc(&ticToc_icdUpdate);
		ticToc_icdUpdate_total += ticToc_icdUpdate;


        /**
         *      Iteration Info
         */
        weightedNormSquared_e = computeSinogramWeightedNormSquared(sino, sino->e);
        weightedNormSquared_y = computeSinogramWeightedNormSquared(sino, sino->vox);
        reconAux.relativeWeightedForwardError = sqrt(weightedNormSquared_e / weightedNormSquared_y);

        normSquared_e = computeNormSquaredFloatArray(&sino->e[0][0][0], N_beta*N_dv*N_dw);
        normSquared_y = computeNormSquaredFloatArray(&sino->vox[0][0][0], N_beta*N_dv*N_dw);
        reconAux.relativeUnweightedForwardError = sqrt(normSquared_e / normSquared_y);


        reconAux.NHICD_isPartialUpdateActive = NHICD_activatePartialUpdate(reconParams, reconAux.relativeWeightedForwardError);


		tic(&ticToc_computeLastChangeThreshold);
		reconAux.lastChangeThreshold = prctile_copyFast(&img->lastChange[0][0][0], N_x*N_y*numZiplines, 100-reconParams->NHICD_percentage, subsampleFactor);
		toc(&ticToc_computeLastChangeThreshold);

		/**
		 *		weightScaler_estimateMode
		 */
		if (strcmp(reconParams->weightScaler_estimateMode,"errorSino") == 0)
			sino->params.weightScaler_value = weightedNormSquared_e;
		else if (strcmp(reconParams->weightScaler_estimateMode,"avgWghtRecon") == 0)
			sino->params.weightScaler_value = computeAvgWghtRecon(img);
		else if (strcmp(reconParams->weightScaler_estimateMode,"None") == 0)
			sino->params.weightScaler_value = reconParams->weightScaler_value;
		else
		{
			fprintf(stderr, "ERROR in MBIR3DCone: can't recongnize weightScaler_estimateMode.\n");
	        exit(-1);
	    }

		tic(&ticToc_computeCost);
        if(reconParams->isComputeCost)
            cost = MAPCost3D(sino, img, reconParams);
		toc(&ticToc_computeCost);
		

		tic(&ticToc_computeRelUpdate);
		relUpdate = computeRelUpdate(&reconAux, reconParams, img);
		toc(&ticToc_computeRelUpdate);

        ratioUpdated = (double) reconAux.NumUpdatedVoxels / numVoxelsInMask;
        reconAux.totalEquits += ratioUpdated;


        if(reconParams->isPhantomReconReference)
        	RRMSE = computeRelativeRMSEFloatArray(&img->vox[0][0][0], &img->phantom[0][0][0], N_x*N_y*N_z);





		tic(&ticToc_IntermediateStoring);
		/**
		 * 		Intermediate storing of binaries
		 */
			/* Error Sino */
			writeSinoData3DCone(pathNames->errSino, (void***)sino->e, &sino->params, "float");
	
			strcpy(tempFName, pathNames->errSino);
			prependToFName(reconParams->downsampleFNamePrefix, tempFName);
			writeDownSampledFloat3D(tempFName, sino->e, N_beta, N_dv, N_dw, reconParams->downsampleFactorSino, 1, 1);

			/* Image */
			writeImageFData3DCone(pathNames->recon, (void***) img->vox, &img->params, 0, "float");
			copyImageF2ROI(img);
			writeImageFData3DCone(pathNames->reconROI, (void***) img->vox_roi, &img->params, 1, "float");

			/* estimateSino */
			floatArray_z_equals_aX_plus_bY(&sino->estimateSino[0][0][0], 1.0, &sino->vox[0][0][0], -1.0, &sino->e[0][0][0], sino->params.N_beta*sino->params.N_dv*sino->params.N_dw);
			writeSinoData3DCone(pathNames->estimateSino, (void***)sino->estimateSino, &sino->params, "float");

			strcpy(tempFName, pathNames->recon);
			prependToFName(reconParams->downsampleFNamePrefix, tempFName);
			writeDownSampledFloat3D(tempFName, img->vox, N_x, N_y, N_z, 1, 1, reconParams->downsampleFactorRecon);

			/* NHICD Arrays */
			applyMask(img->lastChange, N_x, N_y, numZiplines);
			write3DData(pathNames->lastChange, (void***) img->lastChange, N_x, N_y, numZiplines, "float");

			write3DData(pathNames->timeToChange, (void***) img->timeToChange, N_x, N_y, numZiplines, "unsigned char");

		toc(&ticToc_IntermediateStoring);
		toc(&ticToc_iteration);

		/**
		 * 		Check stopping conditions
		 */
        if (itNumber>0)
        {
            if (relUpdate < stopThresholdChange || reconAux.relativeWeightedForwardError < stopThesholdRWFE || reconAux.relativeUnweightedForwardError < stopThesholdRUFE || (reconParams->isPhantomReconReference && RRMSE < stopThesholdRRMSE) )
                stopFlag = 1;
        }



        if (reconParams->verbosity>1)
        {
    		ticToc_logAndDisp(ticToc_randomization,	              "randomization              ");
            ticToc_logAndDisp(ticToc_computeRelUpdate,            "computeRelUpdate           ");
            ticToc_logAndDisp(ticToc_computeLastChangeThreshold,  "computeLastChangeThreshold ");
            ticToc_logAndDisp(ticToc_IntermediateStoring,         "IntermediateStoring        ");
    		ticToc_logAndDisp(ticToc_computeCost, 	              "computeCost                ");
            ticToc_logAndDisp(ticToc_icdUpdate,                   "icdUpdate                  ");
    		ticToc_logAndDisp(ticToc_iteration, 	              "iteration                  ");
    		ticToc_logAndDisp(ticToc_icdUpdate_total, 	          "icdUpdate_total            ");
        }
		dispAndLog_iterationInfo(&reconAux, reconParams, itNumber, MaxIterations, cost, relUpdate, stopThresholdChange, sino->params.weightScaler_value, speedAuxICD.voxelsPerSecond, ticToc_icdUpdate, weightedNormSquared_e, ratioUpdated, RRMSE, stopThesholdRRMSE, reconAux.totalEquits);
	}

	mem_free_1D((void*)reconAux.NHICD_numUpdatedVoxels);
	mem_free_1D((void*)reconAux.NHICD_totalValueChange);
	mem_free_1D((void*)reconAux.NHICD_isPartialZiplineHot);



	
	mem_free_1D((void*)icdInfoArray);
	RandomZiplineAux_free(&img->randomZiplineAux);
	RandomAux_free(&img->randomAux);

	freeParallelAux(&parallelAux);

	toc(&ticToc_all);
	ticToc_logAndDisp(ticToc_all, "MBIR3DCone");

}















