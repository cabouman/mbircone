
#include "recon4D.h"
#include "../CLibraries/allocate.h"
#include "../CLibraries/io4d.h"
#include <math.h>
#include <time.h>
#include <omp.h>



void MBIR_4D(struct Image *img, struct Params *params, struct PathNames *pathNames)
{
	int itNumber = 0, MaxIterations;
	double stopThreshold;
	int j_t, j_x, j_y, j_z, j_txyz;
	int N_t, N_x, N_y, N_z;

	double timer_icd_loop;
	double ticToc_icdUpdate;
	double ticToc_all;
	double ticToc_randomization;
	double ticToc_computeCost;
	double ticToc_IntermediateStoring;
	double ticToc_iteration;

	char stopFlag = 0;

	struct ICDInfo icdInfo;
	struct NeighborhoodInfo neighborhoodInfo;


	/* Iteration statistics */
	double cost = -1.0;
	double normError;
	double TotalValueChange = 0.0, TotalVoxelValue = 0.0;
	int NumUpdatedVoxels = 0.0;
	double relUpdatePercent = 0;
	struct SpeedAuxICD speedAuxICD;

	double tempcost1, tempcost2;


	/* Renaming some variables */
	MaxIterations = params->MaxIterations;
	stopThreshold = params->stopThreshold;
	N_t = img->params.N_t;
	N_x = img->params.N_x;
	N_y = img->params.N_y;
	N_z = img->params.N_z;

	initialize_NeighborhoodInfo( &neighborhoodInfo, img, params);
	/*printNeighborhoodInfo( &neighborhoodInfo );*/

	RandomAux_allocate(&img->randomAux, &img->params);
	RandomAux_Initialize(&img->randomAux, &img->params);


    /**
     * 		Loop initialization
     */
	resetFile(LOG_ICDLOOP);
	resetFile(LOG_STATS);

	timer_reset(&timer_icd_loop);
	tic(&ticToc_all);

	/*debug*/
	if(DEBUG_FLAG) tempcost1 = MAPCost4D(img, params, &neighborhoodInfo);
	if(DEBUG_FLAG) printf("Initial Cost: %f \n",tempcost1);

	for (itNumber = 0; (itNumber <= MaxIterations) && (stopFlag==0); ++itNumber) 
	{

		/**
		 * 		Shuffle the order in which the voxels are updated
		 * 		before every iteration
		 */
		tic(&ticToc_randomization);
		tic(&ticToc_iteration);
		
		/*debug*/
		RandomAux_ShuffleorderTXYZ(&img->randomAux, &img->params);
			
		toc(&ticToc_randomization);


		tic(&ticToc_icdUpdate);
		speedAuxICD_reset(&speedAuxICD);
		resetIterationStats(&TotalValueChange, &TotalVoxelValue, &NumUpdatedVoxels);
		if (itNumber>0)
		{
			/********************************************************************************************/
			/**
			 * 		ICD Single voxel updates
			 */
			/********************************************************************************************/
			for (j_txyz = 0; j_txyz < N_t*N_x*N_y*N_z; ++j_txyz)
			{
				if (timer_hasPassed(&timer_icd_loop, OUTPUT_REFRESH_TIME))
				{
					speedAuxICD_computeSpeed(&speedAuxICD);
					writeICDLoopStatus2File(LOG_ICDLOOP, j_txyz, N_t*N_x*N_y*N_z, itNumber, speedAuxICD.voxelsPerSecond);
				}

				indexExtraction4D(img->randomAux.orderTXYZ[j_txyz], &j_t, N_t, &j_x, N_x, &j_y, N_y, &j_z, N_z);
				prepareICDInfo(j_t, j_x, j_y, j_z, &icdInfo, &neighborhoodInfo, img);

				ICDStep3DCone(img, &icdInfo, params);

				/* Update iteration statistics */
				updateIterationStats(&TotalValueChange, &TotalVoxelValue, &NumUpdatedVoxels, &icdInfo, img);
				speedAuxICD_update(&speedAuxICD, 1);

			}
		}

		/**
		 * 		Iteration Infor
		 */
		speedAuxICD_computeSpeed(&speedAuxICD);
		toc(&ticToc_icdUpdate);
		
		tic(&ticToc_computeCost);
		cost = MAPCost4D(img, params, &neighborhoodInfo);
		toc(&ticToc_computeCost);
		
		relUpdatePercent = computeRelUpdate(NumUpdatedVoxels, TotalValueChange, TotalVoxelValue);
		/**
	 	* 		normError =  1/N ||e||^{2}_{W}
	 	*/
		normError = 2 * MAPCostForward(img, params) / (img->params.N_t * img->params.N_x * img->params.N_y * img->params.N_z) ;



		tic(&ticToc_IntermediateStoring);
		/**
		 * 		Intermediate storing of binaries
		 */
		if(params->isSaveImage) /* Save image to file */
		{
			writeImageData(pathNames, img);
		}
		toc(&ticToc_IntermediateStoring);
		toc(&ticToc_iteration);

		/**
		 * 		Check stopping conditions
		 */
		if (itNumber > 0 && relUpdatePercent < stopThreshold)
		{
			stopFlag = 1;
		}
		ticToc_logAndDisp(ticToc_randomization, "ticToc_randomization");
		ticToc_logAndDisp(ticToc_computeCost, "ticToc_computeCost");
		ticToc_logAndDisp(ticToc_IntermediateStoring, "ticToc_IntermediateStoring");
		ticToc_logAndDisp(ticToc_iteration, "ticToc_iteration");

		dispAndLog_iterationInfo(itNumber, MaxIterations, cost, relUpdatePercent, speedAuxICD.voxelsPerSecond, ticToc_icdUpdate, normError);
	}


	
	RandomAux_free(&img->randomAux);
	free_NeighborhoodInfo( &neighborhoodInfo );

	toc(&ticToc_all);
	ticToc_logAndDisp(ticToc_all, "MBIR_4D");

}

/*debug*/
/*tempcost2 = MAPCost4D(img, params, &neighborhoodInfo);
if(tempcost2>tempcost1){
	printf("Cost inc from %f to %f by %f\n", tempcost1, tempcost2, tempcost2-tempcost1 );
	printf("new val : %f\n",img->denoised[icdInfo.j_t][icdInfo.j_x][icdInfo.j_y][icdInfo.j_z] );
	printICDinfo(&icdInfo);
	printNeighborsInImg(&icdInfo, img);
	writeImageData( pathNames, img );
	exit(-1);
}
tempcost1 = tempcost2;*/













