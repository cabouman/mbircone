
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

#include "icd3d.h"
#include "allocate.h"


void ICDStep3DCone(struct Sino *sino, struct Image *img, struct SysMatrix *A, struct ICDInfo3DCone *icdInfo, struct ReconParams *reconParams, struct ReconAux *reconAux)
{
	/**
	 * 		Updates one voxel. Voxel change is stored in icdInfo->Delta_xj. 
	 */


	/**
	 * 			Compute forward model term of theta1 and theta2:
	 * 		
	 *   	theta1_f = -e^t W A_{*,j}
	 * 		theta2_f = A_{*,j}^t W A _{*,j}
	 */
	computeTheta1Theta2ForwardTerm(sino, A, icdInfo, reconParams);
    /**
     * 			Compute prior model term of theta1 and theta2:
     * 		
     */
    if(reconParams->priorWeight_QGGMRF >= 0)
		computeTheta1Theta2PriorTermQGGMRF(icdInfo, reconParams);

    if(reconParams->priorWeight_proxMap >= 0)
		computeTheta1Theta2PriorTermProxMap(icdInfo, reconParams);

	computeDeltaXjAndUpdate(icdInfo, reconParams, img, reconAux);

	updateErrorSinogram(sino, A, icdInfo);
	
}

void prepareICDInfo(long int j_x, long int j_y, long int j_z, struct ICDInfo3DCone *icdInfo, struct Image *img, struct ReconAux *reconAux, struct ReconParams *reconParams)
{
	icdInfo->old_xj = img->vox[j_x][j_y][j_z];
	icdInfo->proxMapInput_j = img->proxMapInput[j_x][j_y][j_z];
	icdInfo->j_x = j_x;
	icdInfo->j_y = j_y;
	icdInfo->j_z = j_z;
	extractNeighbors(icdInfo, img, reconParams);
	icdInfo->theta1_f = 0;
	icdInfo->theta2_f = 0;
	icdInfo->theta1_p_QGGMRF = 0;
	icdInfo->theta2_p_QGGMRF = 0;
	icdInfo->theta1_p_proxMap = 0;
	icdInfo->theta2_p_proxMap = 0;

}




void extractNeighbors(struct ICDInfo3DCone *icdInfo, struct Image *img, struct ReconParams *reconParams)
{

	long int j_x, j_y, j_z;
	long int N_x, N_y, N_z;
	long int PLx, MIx;
	long int PLy, MIy;
	long int PLz, MIz;

	j_x = icdInfo->j_x;
	j_y = icdInfo->j_y;
	j_z = icdInfo->j_z;

	N_x = img->params.N_x;
	N_y = img->params.N_y;
	N_z = img->params.N_z;


	/**
	 * 		Use reflective boundary conditions to find the indices of the neighbors
	 */
	PLx = (j_x == N_x-1) ? N_x-2 : j_x+1;
	PLy = (j_y == N_y-1) ? N_y-2 : j_y+1;
	PLz = (j_z == N_z-1) ? N_z-2 : j_z+1;

	MIx = (j_x == 0) ? 1 : j_x-1;
	MIy = (j_y == 0) ? 1 : j_y-1;
	MIz = (j_z == 0) ? 1 : j_z-1;

	/**
	 * 		Compute the neighbor pixel values
	 * 		
	 * 			Note that all the pixels of the first half of the arrays
	 * 		 	have a corresponding pixel in the second half of the array
	 * 		 	that is on the spacially opposite side. 
	 * 			 Example: neighborsFace[0] opposite of neighborsFace[3]
	 */
	if (reconParams->bFace>=0)
	{
		/* Face Neighbors (primal) */
		icdInfo->neighborsFace[0] = img->vox[PLx][j_y][j_z];
		icdInfo->neighborsFace[1] = img->vox[j_x][PLy][j_z];
		icdInfo->neighborsFace[2] = img->vox[j_x][j_y][PLz];
		/* Face Neighbors (opposite) */
		icdInfo->neighborsFace[3] = img->vox[MIx][j_y][j_z];
		icdInfo->neighborsFace[4] = img->vox[j_x][MIy][j_z];
		icdInfo->neighborsFace[5] = img->vox[j_x][j_y][MIz];
	}

	if (reconParams->bEdge>=0)
	{
		/* Edge Neighbors (primal) */
		icdInfo->neighborsEdge[ 0] = img->vox[j_x][PLy][PLz];
		icdInfo->neighborsEdge[ 1] = img->vox[j_x][PLy][MIz];
		icdInfo->neighborsEdge[ 2] = img->vox[PLx][j_y][PLz];
		icdInfo->neighborsEdge[ 3] = img->vox[PLx][j_y][MIz];
		icdInfo->neighborsEdge[ 4] = img->vox[PLx][PLy][j_z];
		icdInfo->neighborsEdge[ 5] = img->vox[PLx][MIy][j_z];
		/* Edge Neighbors (opposite) */
		icdInfo->neighborsEdge[ 6] = img->vox[j_x][MIy][MIz];
		icdInfo->neighborsEdge[ 7] = img->vox[j_x][MIy][PLz];
		icdInfo->neighborsEdge[ 8] = img->vox[MIx][j_y][MIz];
		icdInfo->neighborsEdge[ 9] = img->vox[MIx][j_y][PLz];
		icdInfo->neighborsEdge[10] = img->vox[MIx][MIy][j_z];
		icdInfo->neighborsEdge[11] = img->vox[MIx][PLy][j_z];
	}

	if (reconParams->bVertex>=0)
	{
		/* Vertex Neighbors (primal) */
		icdInfo->neighborsVertex[0] = img->vox[PLx][PLy][PLz];
		icdInfo->neighborsVertex[1] = img->vox[PLx][PLy][MIz];
		icdInfo->neighborsVertex[2] = img->vox[PLx][MIy][PLz];
		icdInfo->neighborsVertex[3] = img->vox[PLx][MIy][MIz];
		/* Vertex Neighbors (opposite) */
		icdInfo->neighborsVertex[4] = img->vox[MIx][MIy][MIz];
		icdInfo->neighborsVertex[5] = img->vox[MIx][MIy][PLz];
		icdInfo->neighborsVertex[6] = img->vox[MIx][PLy][MIz];
		icdInfo->neighborsVertex[7] = img->vox[MIx][PLy][PLz];
	}

}


void computeTheta1Theta2ForwardTerm(struct Sino *sino, struct SysMatrix *A, struct ICDInfo3DCone *icdInfo, struct ReconParams *reconParams)
{
	/**
	 * 			Compute forward model term of theta1 and theta2:
	 * 		
	 *   	theta1_f = -e^t W A_{*,j}
	 * 		theta2_f = A_{*,j}^t W A _{*,j}
	 */

	long int i_beta, i_v, i_w;
	long int j_x, j_y, j_z, j_u;
	float B_ij, A_ij;

	j_x = icdInfo->j_x;
	j_y = icdInfo->j_y;
	j_z = icdInfo->j_z;

    for (i_beta = 0; i_beta < sino->params.N_beta; ++i_beta)
    {
    	j_u = A->j_u[j_x][j_y][i_beta];
        for (i_v = A->i_vstart[j_x][j_y][i_beta]; i_v < A->i_vstart[j_x][j_y][i_beta]+A->i_vstride[j_x][j_y][i_beta]; ++i_v)
        {
        	B_ij = A->B_ij_scaler * A->B[j_x][j_y][i_beta*A->i_vstride_max + i_v-A->i_vstart[j_x][j_y][i_beta]];

            for (i_w = A->i_wstart[j_u][j_z]; i_w < A->i_wstart[j_u][j_z]+A->i_wstride[j_u][j_z]; ++i_w)
            {
            	A_ij = B_ij * A->C_ij_scaler * A->C[j_u][j_z*A->i_wstride_max + i_w-A->i_wstart[j_u][j_z]];
            	icdInfo->theta1_f -=		
            							  sino->e[i_beta][i_v][i_w]
            							* sino->wgt[i_beta][i_v][i_w]
            							* A_ij;

            	icdInfo->theta2_f +=	
            							  A_ij
            							* sino->wgt[i_beta][i_v][i_w]
            							* A_ij;
            }
        }
    }

    if(strcmp(reconParams->weightScaler_domain,"spatiallyInvariant") == 0)
    {
	    icdInfo->theta1_f /= sino->params.weightScaler_value;
	    icdInfo->theta2_f /= sino->params.weightScaler_value;
    }
    else
    {
        fprintf(stderr, "ERROR in computeTheta1Theta2ForwardTerm: can't recongnize weightScaler_domain.\n");
        exit(-1);
    }

}

void computeTheta1Theta2PriorTermQGGMRF(struct ICDInfo3DCone *icdInfo, struct ReconParams *reconParams)
{
    /**
     * 			Compute prior model term of theta1 and theta2:
     * 		
	 * 		theta1_p_QGGMRF = 	  sum 		2 b_{j,r} * surrCoeff(x_j - x_r) * (x_j - x_r)
	 * 							{r E ∂j}
	 * 					
	 * 		theta2_p_QGGMRF = 	  sum 		2 b_{j,r} * surrCoeff(x_j - x_r)
	 * 							{r E ∂j}
     */

	int i;
	float delta, surrogateCoeff;
	float sum1Face = 0;
	float sum1Edge = 0;
	float sum1Vertex = 0;
	float sum2Face = 0;
	float sum2Edge = 0;
	float sum2Vertex = 0;

	if (reconParams->bFace>=0)
	{
		for (i = 0; i < 6; ++i)
		{
			delta = icdInfo->old_xj - icdInfo->neighborsFace[i];
			surrogateCoeff = surrogateCoeffQGGMRF(delta, reconParams);
			sum1Face += surrogateCoeff * delta;
			sum2Face += surrogateCoeff;
		}
	}

	if (reconParams->bEdge>=0)
	{
		for (i = 0; i < 12; ++i)
		{
			delta = icdInfo->old_xj - icdInfo->neighborsEdge[i];
			surrogateCoeff = surrogateCoeffQGGMRF(delta, reconParams);
			sum1Edge += surrogateCoeff * delta;
			sum2Edge += surrogateCoeff;
		}
	}

	if (reconParams->bVertex>=0)
	{
		for (i = 0; i < 8; ++i)
		{
			delta = icdInfo->old_xj - icdInfo->neighborsVertex[i];
			surrogateCoeff = surrogateCoeffQGGMRF(delta, reconParams);
			sum1Vertex += surrogateCoeff * delta;
			sum2Vertex += surrogateCoeff;
		}
	}

	icdInfo->theta1_p_QGGMRF =	  2 * reconParams->bFace * sum1Face
								+ 2 * reconParams->bEdge * sum1Edge
								+ 2 * reconParams->bVertex * sum1Vertex;

	icdInfo->theta2_p_QGGMRF = 	  2 * reconParams->bFace * sum2Face
								+ 2 * reconParams->bEdge * sum2Edge
								+ 2 * reconParams->bVertex * sum2Vertex;
}

void computeTheta1Theta2PriorTermProxMap(struct ICDInfo3DCone *icdInfo, struct ReconParams *reconParams)
{
	/**
	 * 		theta1_p_proxMap = 	 (x_j - ~x_j) / (sigma_lambda^2) 
	 * 					
	 * 					
	 * 		theta2_p_proxMap = 	 1 / (sigma_lambda^2) 
	 * 					
	 */
	icdInfo->theta1_p_proxMap = (icdInfo->old_xj - icdInfo->proxMapInput_j) / (reconParams->sigma_lambda * reconParams->sigma_lambda);
	icdInfo->theta2_p_proxMap = 1.0 / (reconParams->sigma_lambda * reconParams->sigma_lambda);
}

float surrogateCoeffQGGMRF(float Delta, struct ReconParams *reconParams)
{
	/**
	 * 				 		 /  rho'(Delta) / (2 Delta) 			if Delta != 0
	 *   surrCoeff(Delta) = {
	 * 				 		 \	rho''(0) / 2 						if Delta = 0
	 */
    float p, q, T, sigmaX, qmp;
    float num, denom, temp;
    
    p = reconParams->p;
    q = reconParams->q;
    T = reconParams->T;
    sigmaX = reconParams->sigmaX;
    qmp = q - p;
    
    if(fabs(Delta) < 1e-5)
    {
    	/**
    	 * 		rho''(0)           1
    	 * 		-------- = -----------------
    	 * 		   2       p sigmaX^q T^(q-p)
    	 */
	    return 1.0 / ( p * pow(sigmaX, q) * pow(T, qmp) );
    }
    else /* Delta != 0 */
    {
	    /**
	     * 		rho'(Delta)   |Delta|^(p-2)  # (q/p + #)
	     * 		----------- = ------------- ------------
	     * 		   2 Delta     2 sigmaX^p    (1 + #)^2
	     *
	     * 		where          | Delta  |^(q-p)
	     * 		          #  = |--------|
	     * 		               |T sigmaX|
	     */
	    temp = pow(fabs(Delta / (T*sigmaX)), qmp);	/* this is the # from above */
	    num = pow(fabs(Delta), p-2) * temp * (q/p + temp);
	    denom = 2 * pow(sigmaX,p) * (1.0 + temp) * (1.0 + temp);
	    
	    return num / denom;
    }

}

void updateErrorSinogram(struct Sino *sino, struct SysMatrix *A, struct ICDInfo3DCone *icdInfo)
{
	/**
	 * 			Update error sinogram
	 * 		
	 * 		e <- e - A_{*,j} * Delta_xj
	 */


	long int i_beta, i_v, i_w;
	long int j_x, j_y, j_z, j_u;
	float B_ij;

	j_x = icdInfo->j_x;
	j_y = icdInfo->j_y;
	j_z = icdInfo->j_z;

    for (i_beta = 0; i_beta < sino->params.N_beta; ++i_beta)
    {
    	j_u = A->j_u[j_x][j_y][i_beta];
        for (i_v = A->i_vstart[j_x][j_y][i_beta]; i_v < A->i_vstart[j_x][j_y][i_beta]+A->i_vstride[j_x][j_y][i_beta]; ++i_v)
        {
        	B_ij = A->B_ij_scaler * A->B[j_x][j_y][i_beta*A->i_vstride_max + i_v-A->i_vstart[j_x][j_y][i_beta]];

            for (i_w = A->i_wstart[j_u][j_z]; i_w < A->i_wstart[j_u][j_z]+A->i_wstride[j_u][j_z]; ++i_w)
            {
            	
            	sino->e[i_beta][i_v][i_w] -= 	
            									  B_ij
            									* A->C_ij_scaler * A->C[j_u][j_z*A->i_wstride_max + i_w-A->i_wstart[j_u][j_z]]
            									* icdInfo->Delta_xj;
            }
        }
    }
}

void updateIterationStats(struct ReconAux *reconAux, struct ICDInfo3DCone *icdInfo, struct Image *img)
{
	reconAux->TotalValueChange += fabs(icdInfo->Delta_xj);
	reconAux->TotalVoxelValue += _MAX_(img->vox[icdInfo->j_x][icdInfo->j_y][icdInfo->j_z], icdInfo->old_xj);
	reconAux->NumUpdatedVoxels++;
}

void resetIterationStats(struct ReconAux *reconAux)
{
	reconAux->TotalValueChange = 0;
	reconAux->TotalVoxelValue = 0;
	reconAux->NumUpdatedVoxels = 0;
}



void RandomAux_ShuffleOrderXYZ(struct RandomAux *aux, struct ImageParams *params)
{
	shuffleLongIntArray(aux->orderXYZ, params->N_x * params->N_y * params->N_z);
}

void indexExtraction3D(long int j_xyz, long int *j_x, long int N_x, long int *j_y, long int N_y, long int *j_z, long int N_z)
{
	/* j_xyz = j_z + N_z j_y + N_z N_y j_x */

	long int j_temp;

	j_temp = j_xyz;					/* Now, j_temp = j_z + N_z j_y + N_z N_y j_x */

	*j_z = j_temp % N_z;
	j_temp = (j_temp-*j_z) / N_z; 	/* Now, j_temp = j_y + N_y j_x */

	*j_y = j_temp % N_y;
	j_temp = (j_temp-*j_y) / N_y;	/* Now, j_temp = j_x */

	*j_x = j_temp;

	return;
}


float MAPCost3D(struct Sino *sino, struct Image *img, struct ReconParams *reconParams)
{
	/**
	 *      Computes MAP cost function
	 */
	float cost;

    // Initialize cost with forward model cost	
    cost = MAPCostForward(sino);

    // if prior is used, add prior cost
    if(reconParams->priorWeight_QGGMRF >= 0)
		cost += MAPCostPrior_QGGMRF(img, reconParams);

    // if proximal map is used, add proximal map cost
    if(reconParams->priorWeight_proxMap >= 0)
		cost += MAPCostPrior_ProxMap(img, reconParams);

	return cost;
}


float MAPCostForward(struct Sino *sino)
{
	/**
	 * 		ForwardCost =  1/2 ||e||^{2}_{W}
	 */
	long int i_beta, i_v, i_w;
	float cost;

    cost = 0;
    for (i_beta = 0; i_beta < sino->params.N_beta; ++i_beta)
    {
        for (i_v = 0; i_v < sino->params.N_dv; ++i_v)
        {
            for (i_w = 0; i_w < sino->params.N_dw; ++i_w)
            {
            	cost +=   sino->e[i_beta][i_v][i_w]
            			* sino->wgt[i_beta][i_v][i_w]
            			* sino->e[i_beta][i_v][i_w];
            }
        }
    }
    return cost / (2.0 * sino->params.weightScaler_value);
}

float MAPCostPrior_QGGMRF(struct Image *img, struct ReconParams *reconParams)
{
	/**
	 *	cost = sum     b_{s,r}  rho(x_s-x_r)
	 * 	      {s,r} E P
	 */
	
	long int j_x, j_y, j_z;
	struct ICDInfo3DCone icdInfo;
	float cost;
	float temp;
    
    cost = 0;
	for (j_x = 0; j_x < img->params.N_x; ++j_x)
	{
		for (j_y = 0; j_y < img->params.N_y; ++j_y)
		for (j_z = 0; j_z < img->params.N_z; ++j_z)
		{
			/**
			 * 		Prepare icdInfo
			 */
			icdInfo.j_x = j_x;
			icdInfo.j_y = j_y;
			icdInfo.j_z = j_z;
			extractNeighbors(&icdInfo, img, reconParams);
			icdInfo.old_xj = img->vox[j_x][j_y][j_z];
			temp = MAPCostPrior_QGGMRFSingleVoxel_HalfNeighborhood(&icdInfo, reconParams);
			cost += temp;
		}
	}
	return cost * reconParams->priorWeight_QGGMRF;
}

float MAPCostPrior_ProxMap(struct Image *img, struct ReconParams *reconParams)
{
    /**
     * 			Compute proximal mapping prior cost
     * 		 		       1         ||        ||2
	 * 		cost += ---------------- || x - x~ ||
	 * 			    2 sigma_lambda^2 ||        ||2
	 * 				
     */
    
    long int j_x, j_y, j_z;
    float cost, diff_voxel;

    cost = 0; 
    for (j_x = 0; j_x < img->params.N_x; ++j_x)
    {
        for (j_y = 0; j_y < img->params.N_y; ++j_y)
        {
            for (j_z = 0; j_z < img->params.N_z; ++j_z)
            {
                diff_voxel = img->vox[j_x][j_y][j_z] - img->proxMapInput[j_x][j_y][j_z];
                cost += diff_voxel*diff_voxel*isInsideMask(j_x, j_y, img->params.N_x, img->params.N_y);
            }
        }
    }
    cost /= 2 * reconParams->sigma_lambda * reconParams->sigma_lambda;
    return cost;
}

float MAPCostPrior_QGGMRFSingleVoxel_HalfNeighborhood(struct ICDInfo3DCone *icdInfo, struct ReconParams *reconParams)
{
    /**
     * 			Compute prior model term of theta1 and theta2:
     * 		
	 * 		cost += 	  sum 		   b_{j,r} * rho(x_j - x_r)
	 * 					{r E ∂j^half}
	 * 				
     */

	int i;
	float sum1Face, sum1Edge, sum1Vertex;

    sum1Face = 0;
    sum1Edge = 0;
    sum1Vertex = 0;
	if (reconParams->bFace>=0)
		for (i = 0; i < 3; ++i) /* Note: only use first half of the neighbors */
			sum1Face += QGGMRFPotential(icdInfo->old_xj - icdInfo->neighborsFace[i], reconParams);

	if (reconParams->bEdge>=0)
		for (i = 0; i < 6; ++i)	/* Note: only use first half of the neighbors */
			sum1Edge += QGGMRFPotential(icdInfo->old_xj - icdInfo->neighborsEdge[i], reconParams);

	if (reconParams->bVertex>=0)
		for (i = 0; i < 4; ++i)	/* Note: only use first half of the neighbors */
			sum1Vertex += QGGMRFPotential(icdInfo->old_xj - icdInfo->neighborsVertex[i], reconParams);

	return    reconParams->bFace * sum1Face
			+ reconParams->bEdge * sum1Edge
			+ reconParams->bVertex * sum1Vertex;

}



/* the potential function of the QGGMRF prior model.  p << q <= 2 */
float QGGMRFPotential(float delta, struct ReconParams *reconParams)
{
    float p, q, T, sigmaX;
    float temp, GGMRF_Pot;
    
    p = reconParams->p;
    q = reconParams->q;
    T = reconParams->T;
    sigmaX = reconParams->sigmaX;
    
    GGMRF_Pot = pow(fabs(delta),p)/(p*pow(sigmaX,p));
    temp = pow(fabs(delta/(T*sigmaX)), q-p);
    return ( GGMRF_Pot * temp/(1.0+temp) );
}

void partialZipline_computeStartStopIndex(long int *j_z_start, long int *j_z_stop, long int indexZiplines, long int numVoxelsPerZipline, long int N_z)
{
    *j_z_start = indexZiplines*numVoxelsPerZipline;
    *j_z_stop  = _MIN_(*j_z_start+numVoxelsPerZipline-1, N_z-1);
}

int partialZipline_computeZiplineIndex(long int j_z, long int numVoxelsPerZipline)
{
	return floor(j_z / numVoxelsPerZipline);
}



void prepareICDInfoRandGroup(long int j_x, long int j_y, struct RandomZiplineAux *randomZiplineAux, struct ICDInfo3DCone *icdInfo, struct Image *img, struct ReconParams *reconParams, struct ReconAux *reconAux)
{
    /* j = j_y + N_y j_x */
    long int j_z, k_M;
    long int j_z_start, j_z_stop;
    long int indexZiplines;

    k_M = 0;

    for (indexZiplines = 0; indexZiplines < reconParams->numZiplines; ++indexZiplines)
    {
	    if (!reconAux->NHICD_isPartialUpdateActive || reconAux->NHICD_isPartialZiplineHot[indexZiplines])
	    {
	    	partialZipline_computeStartStopIndex(&j_z_start, &j_z_stop, indexZiplines, reconParams->numVoxelsPerZipline, img->params.N_z);

		    for (j_z = j_z_start; j_z <= j_z_stop; ++j_z)
		    {
		    	if(randomZiplineAux->groupIndex[j_x][j_y][j_z] == randomZiplineAux->k_G)
		    	{
					prepareICDInfo(j_x, j_y, j_z, &icdInfo[k_M], img, reconAux, reconParams);
			        /* Increment k_M. After loop terminates k_M = No. members */
			        k_M++;
		    	}
		    }
	    }
    	
    }
    randomZiplineAux->N_M = k_M;

}



void computeDeltaXjAndUpdate(struct ICDInfo3DCone *icdInfo, struct ReconParams *reconParams, struct Image *img, struct ReconAux *reconAux)
{
	/**
	 * 			Compute voxel increment Delta_xj.
	 * 		 	Delta_xj >= -x_j accomplishes positivity constraint:
	 * 		
	 * 		Delta_xj = clip{   -theta1/theta2, [-x_j, inf)   }
	 */
	float theta1, theta2;

	theta1 = icdInfo->theta1_f + reconParams->priorWeight_QGGMRF*icdInfo->theta1_p_QGGMRF + reconParams->priorWeight_proxMap*icdInfo->theta1_p_proxMap;
	theta2 = icdInfo->theta2_f + reconParams->priorWeight_QGGMRF*icdInfo->theta2_p_QGGMRF + reconParams->priorWeight_proxMap*icdInfo->theta2_p_proxMap;

	if (theta2 != 0)
	{
		icdInfo->Delta_xj = -theta1/theta2;

		if(reconParams->is_positivity_constraint)
			icdInfo->Delta_xj = _MAX_(icdInfo->Delta_xj, -icdInfo->old_xj);
	}
	else
	{
		icdInfo->Delta_xj = _MAX_(icdInfo->old_xj, 0);
	}

	if(icdInfo->Delta_xj != icdInfo->Delta_xj)
	{
		printf("theta1_f = %e\n", icdInfo->theta1_f);
		printf("theta2_f = %e\n", icdInfo->theta2_f);
		printf("theta1_p_QGGMRF = %e\n", icdInfo->theta1_p_QGGMRF);
		printf("theta2_p_QGGMRF = %e\n", icdInfo->theta2_p_QGGMRF);
		printf("theta1_p_proxMap = %e\n", icdInfo->theta1_p_proxMap);
		printf("theta2_p_proxMap = %e\n", icdInfo->theta2_p_proxMap);
		printf("theta2 = %e\n", theta2);
		printf("theta2 = %e\n", theta2);
		printf("-t1/t2 = %e\n", -theta1/theta2);
		printf("Delta_xj = %e\n", icdInfo->Delta_xj);
		printf("------------------------\n");
	}

	/**
	 * 			Update voxel:
	 * 		
	 * 		x_j <- x_j + Delta_xj
	 */

	img->vox[icdInfo->j_x][icdInfo->j_y][icdInfo->j_z] 			+= icdInfo->Delta_xj;

}

void computeDeltaXjAndUpdateGroup(struct ICDInfo3DCone *icdInfo, struct RandomZiplineAux *randomZiplineAux, struct ReconParams *reconParams, struct Image *img, struct ReconAux *reconAux)
{
	long int N_M, k_M;
	struct ICDInfo3DCone *info;

	N_M = randomZiplineAux->N_M;
	for (k_M = 0; k_M < N_M; ++k_M)
	{
		info = &icdInfo[k_M];
		computeDeltaXjAndUpdate(info, reconParams, img, reconAux);
	}
}


void updateIterationStatsGroup(struct ReconAux *reconAux, struct ICDInfo3DCone *icdInfoArray, struct RandomZiplineAux *randomZiplineAux, struct Image *img, struct ReconParams *reconParams)
{
	long int N_M, k_M;
	float absDelta, totValue;
	struct ICDInfo3DCone *icdInfo;
	long int j_x, j_y, j_z;
	long int indexZiplines;




	j_x = icdInfoArray[0].j_x;
	j_y = icdInfoArray[0].j_y;

	N_M = randomZiplineAux->N_M;
	for (k_M = 0; k_M < N_M; ++k_M)
	{
		icdInfo = &icdInfoArray[k_M];
		j_z = icdInfo->j_z;

		indexZiplines = partialZipline_computeZiplineIndex(j_z, reconParams->numVoxelsPerZipline);

		absDelta = fabs(icdInfo->Delta_xj);
		totValue = _MAX_(img->vox[j_x][j_y][j_z], icdInfo->old_xj);

		reconAux->TotalValueChange 	+= absDelta;
		reconAux->TotalVoxelValue 	+= totValue;
		reconAux->NumUpdatedVoxels++;

		reconAux->NHICD_numUpdatedVoxels[indexZiplines]++;
		reconAux->NHICD_totalValueChange[indexZiplines] += absDelta;

	}
}


void disp_iterationInfo(struct ReconAux *reconAux, struct ReconParams *reconParams, int itNumber, int MaxIterations, float cost, float relUpdate, float stopThresholdChange, float weightScaler_value, float voxelsPerSecond, float ticToc_iteration, float weightedNormSquared_e, float ratioUpdated, float totalEquits)
{
	printf("************************** Iteration %-2d (max. %d) **************************\n", itNumber, MaxIterations);
	printf("*  Cost                   = %-10.10e\n", cost);
	printf("*  Rel. Update            = %-10.10e %% (threshold = %-10.10e %%)\n", relUpdate*100, stopThresholdChange*100);
	printf("*  RWFE = ||e||_W/||y||_W = %-10.10e %% (threshold = %-10.10e %%)\n", reconAux->relativeWeightedForwardError*100, reconParams->stopThesholdRWFE_pct);
	printf("*  RUFE = ||e|| / ||y||   = %-10.10e %% (threshold = %-10.10e %%)\n", reconAux->relativeUnweightedForwardError*100, reconParams->stopThesholdRUFE_pct);
	printf("* ----------------------------------------------------------------------------\n");
	printf("*  1/M ||e||^2_W          = %-10.10e = 1/%-10.10f\n", weightedNormSquared_e, 1/weightedNormSquared_e);
	printf("*  weightScaler_value     = %-10.10e = 1/%-10.10f\n", weightScaler_value, 1/weightScaler_value);
	printf("* ----------------------------------------------------------------------------\n");
	printf("*  voxelsPerSecond        = %-10.10e \n", voxelsPerSecond);
	printf("*  time icd update        = %-10.10e s\n", ticToc_iteration);
	printf("*  ratioUpdated           = %-10.10e %%\n", ratioUpdated*100);
	printf("*  totalEquits            = %-10.10e \n", totalEquits);
	printf("******************************************************************************\n\n");
}

float computeRelUpdate(struct ReconAux *reconAux, struct ReconParams *reconParams, struct Image *img)
{
	float relUpdate;
	float AvgValueChange, AvgVoxelValue;
	float scaler;
	int subsampleFactor = 10; /* when chosen 1 this is completely accurate. User can mess with this to some extend*/

	if(reconAux->NumUpdatedVoxels>0)
	{
		AvgValueChange = reconAux->TotalValueChange /  reconAux->NumUpdatedVoxels;
		AvgVoxelValue = reconAux->TotalVoxelValue /  reconAux->NumUpdatedVoxels;
	}
	else
	{
		AvgValueChange = 0;
		AvgVoxelValue = 0;
	}

	if(AvgVoxelValue>0)
	{
		/* [relativeChangeMode] 'meanImage' or 'fixedScaler' or 'percentile' */
		if (strcmp(reconParams->relativeChangeMode, "meanImage")==0)
		{
			relUpdate = AvgValueChange / AvgVoxelValue;
		}
		else if (strcmp(reconParams->relativeChangeMode, "fixedScaler")==0)
		{
			relUpdate = AvgValueChange / reconParams->relativeChangeScaler;
		}
		else if (strcmp(reconParams->relativeChangeMode, "percentile")==0)
		{
			scaler = prctile_copyFast(&img->vox[0][0][0], img->params.N_x*img->params.N_y*img->params.N_z,  reconParams->relativeChangePercentile, subsampleFactor);
			relUpdate = AvgValueChange / scaler;
		}
		else
		{
			printf("Error: relativeChangeMode unknown\n");
			exit(-1);
		}
	}
	else
	{
		relUpdate = 0;
	}

	return relUpdate;
}


/* * * * * * * * * * * * parallel * * * * * * * * * * * * **/
void prepareParallelAux(struct ParallelAux *parallelAux, long int N_M_max)
{
	int numThreads;
	#pragma omp parallel
	{
		#pragma omp master
		{
			parallelAux->numThreads = numThreads = omp_get_num_threads();
		}
	}
	parallelAux->N_M_max = N_M_max;

	parallelAux->partialTheta = (struct PartialTheta**) mem_alloc_2D(numThreads, N_M_max, sizeof(struct PartialTheta));

	parallelAux->j_u = mem_alloc_1D(numThreads, sizeof(long int));
	parallelAux->i_v = mem_alloc_1D(numThreads, sizeof(long int));
	parallelAux->B_ij = mem_alloc_1D(numThreads, sizeof(float));
	parallelAux->k_M = mem_alloc_1D(numThreads, sizeof(long int));
	parallelAux->j_z = mem_alloc_1D(numThreads, sizeof(long int));
	parallelAux->i_w = mem_alloc_1D(numThreads, sizeof(long int));
	parallelAux->A_ij = mem_alloc_1D(numThreads, sizeof(float));

}

void freeParallelAux(struct ParallelAux *parallelAux)
{
	mem_free_2D((void**)parallelAux->partialTheta);

	mem_free_1D((void*)parallelAux->j_u);
	mem_free_1D((void*)parallelAux->i_v);
	mem_free_1D((void*)parallelAux->B_ij);
	mem_free_1D((void*)parallelAux->k_M);
	mem_free_1D((void*)parallelAux->j_z);
	mem_free_1D((void*)parallelAux->i_w);
	mem_free_1D((void*)parallelAux->A_ij);

}

void ICDStep3DConeGroup(struct Sino *sino, struct Image *img, struct SysMatrix *A, struct ICDInfo3DCone *icdInfo, struct ReconParams *reconParams, struct RandomZiplineAux *randomZiplineAux, struct ParallelAux *parallelAux, struct ReconAux *reconAux)
{
	if (randomZiplineAux->N_M>0)
	{
		computeTheta1Theta2ForwardTermGroup(sino, A, icdInfo, randomZiplineAux, parallelAux, reconParams);

	    if(reconParams->priorWeight_QGGMRF >= 0)
			computeTheta1Theta2PriorTermQGGMRFGroup(icdInfo, reconParams, randomZiplineAux);

	    if(reconParams->priorWeight_proxMap >= 0)
			computeTheta1Theta2PriorTermProxMapGroup(icdInfo, reconParams, randomZiplineAux);

	    computeDeltaXjAndUpdateGroup(icdInfo, randomZiplineAux, reconParams, img, reconAux);

		updateErrorSinogramGroup(sino, A, icdInfo, randomZiplineAux);
	}

}

void computeTheta1Theta2ForwardTermGroup(struct Sino *sino, struct SysMatrix *A, struct ICDInfo3DCone *icdInfo, struct RandomZiplineAux *randomZiplineAux, struct ParallelAux *parallelAux, struct ReconParams *reconParams)
{
	/**
	 * 			Compute forward model term of theta1 and theta2 for all members:
	 * 		
	 *   	theta1_f = -e^t W A_{*,j}
	 * 		theta2_f = A_{*,j}^t W A _{*,j}
	 */

	long int i_beta, i_v, i_w;
	long int j_x, j_y, j_z, j_u;
	float B_ij, A_ij;
	long int N_M, k_M;
	int threadID;

	N_M = randomZiplineAux->N_M;
	j_x = (icdInfo[0]).j_x;
	j_y = (icdInfo[0]).j_y;


	for (threadID = 0; threadID < parallelAux->numThreads; ++threadID)
	{
		for (k_M = 0; k_M < N_M; ++k_M)
		{
			parallelAux->partialTheta[threadID][k_M].t1 = 0;
			parallelAux->partialTheta[threadID][k_M].t2 = 0;
		}
	}

	#pragma omp parallel private(threadID, j_u, i_v, B_ij, k_M, j_z, i_w, A_ij)
	{
		threadID = omp_get_thread_num();

		#pragma omp for
	    for (i_beta = 0; i_beta < sino->params.N_beta; ++i_beta)
	    {
	    	j_u = A->j_u[j_x][j_y][i_beta];
	        for (i_v = A->i_vstart[j_x][j_y][i_beta]; i_v < A->i_vstart[j_x][j_y][i_beta]+A->i_vstride[j_x][j_y][i_beta]; ++i_v)
	        {
	        	B_ij = A->B_ij_scaler * A->B[j_x][j_y][i_beta*A->i_vstride_max + i_v-A->i_vstart[j_x][j_y][i_beta]];


	        	/* Loop through all the members along zip line */
	        	for (k_M = 0; k_M < N_M; ++k_M)
	        	{
	        		j_z = icdInfo[k_M].j_z;
		            for (i_w = A->i_wstart[j_u][j_z]; i_w < A->i_wstart[j_u][j_z]+A->i_wstride[j_u][j_z]; ++i_w)
		            {
		            	A_ij = B_ij * A->C_ij_scaler * A->C[j_u][j_z*A->i_wstride_max + i_w-A->i_wstart[j_u][j_z]];
		            	
		            	parallelAux->partialTheta[threadID][k_M].t1 -=	 
		            													  sino->e[i_beta][i_v][i_w]
		            													* sino->wgt[i_beta][i_v][i_w]
		            													* A_ij;

		            	parallelAux->partialTheta[threadID][k_M].t2 +=	
		            													  A_ij
		            													* sino->wgt[i_beta][i_v][i_w]
		            													* A_ij;

		            }
	        	}
	        }
	    }
	}


	for (threadID = 0; threadID < parallelAux->numThreads; ++threadID)
	{
		for (k_M = 0; k_M < N_M; ++k_M)
		{
			icdInfo[k_M].theta1_f += parallelAux->partialTheta[threadID][k_M].t1;
			icdInfo[k_M].theta2_f += parallelAux->partialTheta[threadID][k_M].t2;
		}
	}

    if(strcmp(reconParams->weightScaler_domain,"spatiallyInvariant") == 0)
    {
        for (k_M = 0; k_M < N_M; ++k_M)
        {
            icdInfo[k_M].theta1_f /= sino->params.weightScaler_value;
            icdInfo[k_M].theta2_f /= sino->params.weightScaler_value;
        }
    }
    else
    {
        fprintf(stderr, "ERROR in computeTheta1Theta2ForwardTerm: can't recongnize weightScaler_domain.\n");
        exit(-1);
    }

}

void computeTheta1Theta2PriorTermQGGMRFGroup(struct ICDInfo3DCone *icdInfo, struct ReconParams *reconParams, struct RandomZiplineAux *randomZiplineAux)
{
	long int N_M, k_M;

	N_M = randomZiplineAux->N_M;
	#pragma omp parallel for
	for (k_M = 0; k_M < N_M; ++k_M)
	{
		computeTheta1Theta2PriorTermQGGMRF(&icdInfo[k_M], reconParams);
	}
}

void updateErrorSinogramGroup(struct Sino *sino, struct SysMatrix *A, struct ICDInfo3DCone *icdInfo, struct RandomZiplineAux *randomZiplineAux)
{
	/**
	 * 			Update error sinogram
	 * 		
	 * 		e <- e - A_{*,j} * Delta_xj
	 */
	
	long int N_M, k_M;


	long int i_beta, i_v, i_w;
	long int j_x, j_y, j_z, j_u;
	float B_ij;

	N_M = randomZiplineAux->N_M;
	j_x = icdInfo[0].j_x;
	j_y = icdInfo[0].j_y;

	#pragma omp parallel for private(j_u, i_v, B_ij, k_M, j_z, i_w)
    for (i_beta = 0; i_beta < sino->params.N_beta; ++i_beta)
    {
    	j_u = A->j_u[j_x][j_y][i_beta];
        for (i_v = A->i_vstart[j_x][j_y][i_beta]; i_v < A->i_vstart[j_x][j_y][i_beta]+A->i_vstride[j_x][j_y][i_beta]; ++i_v)
        {
        	B_ij = A->B_ij_scaler * A->B[j_x][j_y][i_beta*A->i_vstride_max + i_v-A->i_vstart[j_x][j_y][i_beta]];

        	for (k_M = 0; k_M < N_M; ++k_M)
        	{
        		j_z = icdInfo[k_M].j_z;
	            for (i_w = A->i_wstart[j_u][j_z]; i_w < A->i_wstart[j_u][j_z]+A->i_wstride[j_u][j_z]; ++i_w)
	            {
	            	
	            	sino->e[i_beta][i_v][i_w] -= 	
	            									  B_ij
	            									* A->C_ij_scaler * A->C[j_u][j_z*A->i_wstride_max + i_w-A->i_wstart[j_u][j_z]]
	            									* icdInfo[k_M].Delta_xj;
	            }
        	}
        }
    }
}


void computeTheta1Theta2PriorTermProxMapGroup(struct ICDInfo3DCone *icdInfo, struct ReconParams *reconParams, struct RandomZiplineAux *randomZiplineAux)
{
	long int N_M, k_M;

	N_M = randomZiplineAux->N_M;

	for (k_M = 0; k_M < N_M; ++k_M)
	{
		icdInfo[k_M].theta1_p_proxMap = (icdInfo[k_M].old_xj - icdInfo[k_M].proxMapInput_j) / (reconParams->sigma_lambda * reconParams->sigma_lambda);
		icdInfo[k_M].theta2_p_proxMap = 1.0 / (reconParams->sigma_lambda * reconParams->sigma_lambda);
	}

}


/* * * * * * * * * * * * time aux ICD * * * * * * * * * * * * **/

void speedAuxICD_reset(struct SpeedAuxICD *speedAuxICD)
{
	speedAuxICD->numberUpdatedVoxels = 0;
	speedAuxICD->tic = omp_get_wtime();
	speedAuxICD->toc = -1.0;
	speedAuxICD->voxelsPerSecond = -1.0;
}

void speedAuxICD_update(struct SpeedAuxICD *speedAuxICD, long int incrementNumber)
{
	speedAuxICD->numberUpdatedVoxels += incrementNumber;

}
 
void speedAuxICD_computeSpeed(struct SpeedAuxICD *speedAuxICD)
{
	if (speedAuxICD->numberUpdatedVoxels > 0)
	{
		speedAuxICD->toc = omp_get_wtime();
		speedAuxICD->voxelsPerSecond = ((float)speedAuxICD->numberUpdatedVoxels) / (speedAuxICD->toc - speedAuxICD->tic);
	}
	else
	{
		speedAuxICD->voxelsPerSecond = 0;
	}
}


/* * * * * * * * * * * * NHICD * * * * * * * * * * * * **/

int NHICD_isVoxelHot(struct ReconParams *reconParams, struct Image *img, long int j_x, long int j_y, long int j_z, float lastChangeThreshold)
{
    if(img->lastChange[j_x][j_y][j_z] > lastChangeThreshold)
        return 1;

    if(bernoulli(reconParams->NHICD_random/100)==1)
    	return 1;

    return 0;
}

int NHICD_activatePartialUpdate(struct ReconParams *reconParams, float relativeWeightedForwardError)
{
	if (relativeWeightedForwardError*100<reconParams->NHICD_ThresholdAllVoxels_ErrorPercent && strcmp(reconParams->NHICD_Mode, "off")!=0)
		return 1;
	else
		return 0;
}

int NHICD_checkPartialZiplineHot(struct ReconAux *reconAux, long int j_x, long int j_y, long int indexZiplines, struct Image *img)
{
	if (reconAux->NHICD_isPartialUpdateActive)
	{
		if (img->lastChange[j_x][j_y][indexZiplines]>=reconAux->lastChangeThreshold || img->timeToChange[j_x][j_y][indexZiplines]==0)
		{
			return 1;
		}
		else
		{
			img->timeToChange[j_x][j_y][indexZiplines] = _MAX_(img->timeToChange[j_x][j_y][indexZiplines]-1, 0);
			return 0;
		}
	}
	else
	{
		return 1;
	}


}

void NHICD_checkPartialZiplinesHot(struct ReconAux *reconAux, long int j_x, long int j_y, struct ReconParams *reconParams, struct Image *img)
{
	long int indexZiplines;

	for (indexZiplines = 0; indexZiplines < reconParams->numZiplines; ++indexZiplines)
	{
		reconAux->NHICD_isPartialZiplineHot[indexZiplines] =  NHICD_checkPartialZiplineHot(reconAux, j_x, j_y, indexZiplines, img);
		reconAux->NHICD_numUpdatedVoxels[indexZiplines] = 0;
		reconAux->NHICD_totalValueChange[indexZiplines] = 0;
	}
}

void updateNHICDStats(struct ReconAux *reconAux, long int j_x, long int j_y, struct Image *img, struct ReconParams *reconParams)
{
	long int jj_x, jj_y, jj_x_min, jj_y_min, jj_x_max, jj_y_max;
	float avgChange;
	float mean_timeToChange;
	long int sigma_timeToChange;
	long int indexZiplines;
	float w_self = 1;
	float w_past = 0.5;
	float w_neighbors = 0.5;



	mean_timeToChange = 100.0/reconParams->NHICD_random-1;
	sigma_timeToChange = round(mean_timeToChange*0.5);

	jj_x_min = _MAX_(j_x-1, 0);
	jj_y_min = _MAX_(j_y-1, 0);
	jj_x_max = _MIN_(j_x+1, img->params.N_x-1);
	jj_y_max = _MIN_(j_y+1, img->params.N_y-1);

	for (indexZiplines = 0; indexZiplines < reconParams->numZiplines; ++indexZiplines)
	{
		if (reconAux->NHICD_isPartialZiplineHot[indexZiplines])
		{
			avgChange = reconAux->NHICD_numUpdatedVoxels[indexZiplines] > 0 ? reconAux->NHICD_totalValueChange[indexZiplines]/reconAux->NHICD_numUpdatedVoxels[indexZiplines] : 0;
			img->lastChange[j_x][j_y][indexZiplines] = w_past * img->lastChange[j_x][j_y][indexZiplines] + w_self * avgChange;
			for (jj_x = jj_x_min; jj_x <= jj_x_max; ++jj_x)
			{
				for (jj_y = jj_y_min; jj_y <= jj_y_max; ++jj_y)
				{
					img->lastChange[jj_x][jj_y][indexZiplines] += w_neighbors * reconAux->NHICD_neighborFilter[1+jj_x-j_x][1+jj_y-j_y] * avgChange;
				}
			}


			img->timeToChange[j_x][j_y][indexZiplines] = almostUniformIntegerRV(mean_timeToChange, sigma_timeToChange);
		}

	}





}













