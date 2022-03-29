
#include <math.h>
#include <time.h>
#include <omp.h>
#include "recon3DCone.h"
#include "allocate.h"

void MBIR3DCone(struct Image *img, struct Sino *sino, struct ReconParams *reconParams, struct SysMatrix *A)
{
	int itNumber = 0, MaxIterations;
    float stopThresholdChange;
    float stopThesholdRWFE, stopThesholdRUFE;
	long int j_xy, j_x, j_y, j_z;
	long int j_xyz;
	long int N_x, N_y, N_z;
	long int N_beta, N_dv, N_dw;
	long int k_G, N_G;
	long int numZiplines;
	long int numVoxelsInMask;
    float ratioUpdated;
    float relUpdate;

	float timer_icd_loop;
	float ticToc_icdUpdate;
	float ticToc_icdUpdate_total;
	float ticToc_all;
	float ticToc_randomization;
	float ticToc_computeCost;
	float ticToc_computeRelUpdate;
	float ticToc_iteration;
	float ticToc_computeLastChangeThreshold;

	char stopFlag = 0;

	struct ICDInfo3DCone *icdInfoArray;			/* Only used when using zip line option*/
	struct ICDInfo3DCone icdInfo;				/* Only used when not using zip line option*/
	struct ParallelAux parallelAux;

	/* Hardcoded stuff */
	int subsampleFactor = 10;


	/* Iteration statistics */
	float cost = -1.0;
	float weightedNormSquared_e, weightedNormSquared_y;
	float normSquared_e, normSquared_y;


	struct SpeedAuxICD speedAuxICD;
    struct ReconAux reconAux;

	/* Renaming some variables */
	MaxIterations = reconParams->MaxIterations;
    stopThresholdChange = reconParams->stopThresholdChange_pct/100.0;
    stopThesholdRWFE = reconParams->stopThesholdRWFE_pct/100.0;
    stopThesholdRUFE = reconParams->stopThesholdRUFE_pct/100.0;
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
	reconAux.NHICD_totalValueChange = (float*) malloc(numZiplines*sizeof(float));
	reconAux.NHICD_isPartialZiplineHot = (int*) malloc(numZiplines*sizeof(int));



	srand(0);

    numVoxelsInMask = computeNumVoxelsInImageMask(img);

    if (reconParams->verbosity>0){
		printImgParams(&img->params);
		printSinoParams(&sino->params);
		printReconParams(reconParams);
	}


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
	/*omp_set_num_threads(reconParams->numThreads);*/
	prepareParallelAux(&parallelAux, reconAux.N_M_max);


    /**
     * 		Loop initialization
     */
	icdInfoArray = mem_alloc_1D(reconAux.N_M_max, sizeof(struct ICDInfo3DCone));

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
		//printf("\r                                                                                    \r");
		speedAuxICD_computeSpeed(&speedAuxICD);
		toc(&ticToc_icdUpdate);
		ticToc_icdUpdate_total += ticToc_icdUpdate;


        /**
         *      Iteration Info
         */
        weightedNormSquared_e = computeSinogramWeightedNormSquared(sino, sino->e);
        weightedNormSquared_y = computeSinogramWeightedNormSquared(sino, sino->vox);
        if (weightedNormSquared_y>0.0) 
            reconAux.relativeWeightedForwardError = sqrt(weightedNormSquared_e / weightedNormSquared_y);
        else
            reconAux.relativeWeightedForwardError = sqrt(weightedNormSquared_e);

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

        ratioUpdated = (float) reconAux.NumUpdatedVoxels / numVoxelsInMask;
        reconAux.totalEquits += ratioUpdated;

		/* NHICD Arrays */
		applyMask(img->lastChange, N_x, N_y, numZiplines);

		toc(&ticToc_iteration);

		/**
		 * 		Check stopping conditions
		 */
        if (itNumber>0)
        {
            if (relUpdate < stopThresholdChange || reconAux.relativeWeightedForwardError < stopThesholdRWFE || reconAux.relativeUnweightedForwardError < stopThesholdRUFE )
                stopFlag = 1;
        }



        if (reconParams->verbosity>1)
        {
    		ticTocDisp(ticToc_randomization,	              "randomization              ");
            ticTocDisp(ticToc_computeRelUpdate,            "computeRelUpdate           ");
            ticTocDisp(ticToc_computeLastChangeThreshold,  "computeLastChangeThreshold ");
    		ticTocDisp(ticToc_computeCost, 	              "computeCost                ");
            ticTocDisp(ticToc_icdUpdate,                   "icdUpdate                  ");
    		ticTocDisp(ticToc_iteration, 	              "iteration                  ");
    		ticTocDisp(ticToc_icdUpdate_total, 	          "icdUpdate_total            ");
        }
        if (reconParams->verbosity>0)
			disp_iterationInfo(&reconAux, reconParams, itNumber, MaxIterations, cost, relUpdate, stopThresholdChange, sino->params.weightScaler_value, speedAuxICD.voxelsPerSecond, ticToc_icdUpdate, weightedNormSquared_e, ratioUpdated, reconAux.totalEquits);
	}

	mem_free_1D((void*)reconAux.NHICD_numUpdatedVoxels);
	mem_free_1D((void*)reconAux.NHICD_totalValueChange);
	mem_free_1D((void*)reconAux.NHICD_isPartialZiplineHot);



	
	mem_free_1D((void*)icdInfoArray);
	RandomZiplineAux_free(&img->randomZiplineAux);
	RandomAux_free(&img->randomAux);

	freeParallelAux(&parallelAux);



	if (reconParams->verbosity>0){
		toc(&ticToc_all);
		ticTocDisp(ticToc_all, "MBIR3DCone");
	}
	

}















