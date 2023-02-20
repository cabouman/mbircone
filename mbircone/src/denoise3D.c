
#include <math.h>
#include <time.h>
#include <omp.h>
#include "denoise3D.h"
#include "allocate.h"

void MBIR3DDenoise(struct Image *img, struct Image *err_img, struct ReconParams *reconParams)
{
    int itNumber = 0, MaxIterations;
    float stopThresholdChange;
    long int j_x, j_y, j_z;
    long int j_xyz;
    long int N_x, N_y, N_z;
    float relUpdate;

    float timer_icd_loop;
    float ticToc_icdUpdate;
    float ticToc_icdUpdate_total;
    float ticToc_all;
    float ticToc_iteration;

    char stopFlag = 0;
    struct ICDInfo3DCone icdInfo;                /* Only used when not using zip line option*/

    /* Iteration statistics */
    float cost;
    // ||y||^2

    struct ReconAux reconAux;

    /* Renaming some variables */
    MaxIterations = reconParams->MaxIterations;
    stopThresholdChange = reconParams->stopThresholdChange_pct/100.0;
    N_x = img->params.N_x;
    N_y = img->params.N_y;
    N_z = img->params.N_z;

    /* reconAux*/
    reconAux.N_M_max = ceil(img->params.N_z/2.0);
    reconAux.TotalValueChange = 0.0;
    reconAux.TotalVoxelValue = 0.0;
    reconAux.NumUpdatedVoxels = 0.0;

    srand(0);

    if (reconParams->verbosity>0){
        printDenoiseParams(&img->params, reconParams);
    }


    /**
     *         Random Auxiliary
     */
    RandomAux_allocate(&img->randomAux, &img->params);
    RandomAux_Initialize(&img->randomAux, &img->params);

    timer_reset(&timer_icd_loop);
    tic(&ticToc_all);
    ticToc_icdUpdate_total = 0;
    for (itNumber = 0; (itNumber <= MaxIterations) && (stopFlag==0); ++itNumber)
    {

        tic(&ticToc_iteration);
        /**
         *         Shuffle the order in which the voxels are updated before every iteration
         */
        RandomAux_ShuffleOrderXYZ(&img->randomAux, &img->params);

        tic(&ticToc_icdUpdate);
        resetIterationStats(&reconAux);
        if (itNumber>0)
        {

            /********************************************************************************************/
            /**
             *         ICD Single voxel updates
             */
            /********************************************************************************************/
            for (j_xyz = 0; j_xyz < N_x*N_y*N_z; ++j_xyz)
            {
                /**
                 *         Prepare icdInfo
                 */
                indexExtraction3D(img->randomAux.orderXYZ[j_xyz], &j_x, N_x, &j_y, N_y, &j_z, N_z);
                prepareICDInfo(j_x, j_y, j_z, &icdInfo, img, &reconAux, reconParams);

                /**
                 *         ICD update of one voxel
                 */
                ICDStep3DDenoise(img, err_img, &icdInfo, reconParams);
                /**
                 *         Update iteration statistics
                 */
                updateIterationStats(&reconAux, &icdInfo, img);
            }
        }
        toc(&ticToc_icdUpdate);
        ticToc_icdUpdate_total += ticToc_icdUpdate;

        /**
         *      Iteration Info
         */

        cost = MAPCost3DDenoise(img, err_img, reconParams);

        relUpdate = computeRelUpdate(&reconAux, reconParams, img);

        toc(&ticToc_iteration);

        /**
         *         Check stopping conditions
         */
        if (itNumber>0)
        {
            if (relUpdate < stopThresholdChange)
                stopFlag = 1;
        }



        if (reconParams->verbosity>0)
            disp_iterationInfo_denoise(itNumber, MaxIterations, cost, relUpdate, stopThresholdChange, reconParams->weightScaler_value, ticToc_icdUpdate);
    }

    RandomAux_free(&img->randomAux);

    if (reconParams->verbosity>0){
        toc(&ticToc_all);
        ticTocDisp(ticToc_all, "MBIR3DDenoise");
    }


}


