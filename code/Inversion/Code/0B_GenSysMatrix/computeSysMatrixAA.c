
#include "computeSysMatrixAA.h"
#include <limits.h>

void computeSysMatrixAA(struct SinoParams *sinoParams, struct ImageFParams *imgParams, struct SysMatrix *A, struct ReconParams *reconParams, struct ViewAngleList *viewAngleList, int hiResFactor)
{
	struct SysMatrix A_hiRes;
	struct ImageFParams imgParams_hiRes;
    int j_x, j_y, i_beta, j_u, j_z;
    int i_vstart_min = INT_MAX, i_wstart_min = INT_MAX;
    int i_vstride_max = INT_MIN, i_wstride_max = INT_MIN;
    int j_x_sub, j_y_sub, j_u_sub, j_z_sub;
    int j_x_HR, j_y_HR, j_u_HR, j_z_HR;


    int i_v, i_w;
    int j_u_min, j_u_max;

    logAndDisp_message(LOG_PROGRESS, "Compute AA System Matrix ...\n");

	/**
	 * 		params
	 */
	imgParams_hiRes = *imgParams;
	imgParams_hiRes.Delta_xy /= hiResFactor;
	imgParams_hiRes.Delta_z  /= hiResFactor;
	imgParams_hiRes.N_x  = (imgParams_hiRes.N_x+1) * hiResFactor;
	imgParams_hiRes.N_y  = (imgParams_hiRes.N_y+1) * hiResFactor;;
	imgParams_hiRes.N_z  *= hiResFactor;

	/**
	 * 		Compute High resolution AMatrix
	 */
	computeSysMatrix(sinoParams, &imgParams_hiRes, &A_hiRes, reconParams, viewAngleList);

	logAndDisp_message(LOG_PROGRESS, "Parameters for A_hiRes:\n");
	printSysMatrixParams(&A_hiRes);

	/**
	 * 		Transform high resolution matrix into low resolution anti aliasing matrix
	 */
    A->i_vstart = 	(int ***)   	mem_alloc_3D(imgParams->N_x, imgParams->N_y,  sinoParams->N_beta, sizeof(int));
    A->i_vstride = 	(int ***)   	mem_alloc_3D(imgParams->N_x, imgParams->N_y,  sinoParams->N_beta, sizeof(int));
    A->u_v = 		(float ***)		mem_alloc_3D(imgParams->N_x, imgParams->N_y,  sinoParams->N_beta, sizeof(float));


	/**
	 * 		Find Matrix parameters
	 */
	A->i_vstride_max = 0;
	A->u_0 = 1e20;
	A->u_1 = -1e20;
	for (j_x = 0; j_x < imgParams->N_x; ++j_x)
	{
		for (j_y = 0; j_y < imgParams->N_y; ++j_y)
		{
			for (i_beta = 0; i_beta < sinoParams->N_beta; ++i_beta)
			{
				i_vstart_min = INT_MAX;
				i_vstride_max = INT_MIN;

				A->u_v[j_x][j_y][i_beta] = 0;
				for (j_x_sub = 0; j_x_sub < hiResFactor; ++j_x_sub)
				{
					for (j_y_sub = 0; j_y_sub < hiResFactor; ++j_y_sub)
					{

						j_x_HR = j_x * hiResFactor + j_x_sub;
						j_y_HR = j_y * hiResFactor + j_y_sub;

						i_vstart_min = _MIN_(A_hiRes.i_vstart[j_x_HR][j_y_HR][i_beta], i_vstart_min);
						i_vstride_max  = _MAX_(A_hiRes.i_vstride [j_x_HR][j_y_HR][i_beta], i_vstride_max);

						A->u_v[j_x][j_y][i_beta] += A_hiRes.u_v[j_x_HR][j_y_HR][i_beta];


					}
				}

				
				A->i_vstride_max = _MAX_(i_vstride_max - i_vstart_min + 1, A->i_vstride_max);
				A->i_vstart[j_x][j_y][i_beta] = i_vstart_min;
				A->i_vstride[j_x][j_y][i_beta]  = i_vstride_max;

				A->u_v[j_x][j_y][i_beta] /= hiResFactor*hiResFactor;
				A->u_0 = _MIN_(A->u_v[j_x][j_y][i_beta] - imgParams->Delta_xy/2, A->u_0);
				A->u_1 = _MAX_(A->u_v[j_x][j_y][i_beta] - imgParams->Delta_xy/2, A->u_1);

			}
		}
	}

	A->Delta_u = A_hiRes.Delta_u * hiResFactor;
	A->N_u = ceil((A->u_1 - A->u_0) / A->Delta_u) + 1;
	A->u_1 = A->u_0 + A->N_u * A->Delta_u; 	/* Find most accurate value of u_1 */


    A->i_wstart = 	(int **) 		mem_alloc_2D(A->N_u, imgParams->N_z, sizeof(int));
    A->i_wstride  = 	(int **) 		mem_alloc_2D(A->N_u, imgParams->N_z, sizeof(int));

    A->i_wstride_max = 0;
	for (j_u = 0; j_u < A->N_u; ++j_u)
	{
		for (j_z = 0; j_z < imgParams->N_z; ++j_z)
		{
			i_wstart_min = INT_MAX;
			i_wstride_max = INT_MIN;
			for (j_u_sub = 0; j_u_sub < hiResFactor; ++j_u_sub)
			{
				for (j_z_sub = 0; j_z_sub < hiResFactor; ++j_z_sub)
				{
					j_u_HR = j_u * hiResFactor + (int) round(j_u_sub * imgParams->Delta_xy / (A_hiRes.Delta_u * hiResFactor));
					j_z_HR = j_z * hiResFactor + j_z_sub;

					i_wstart_min = _MIN_(A_hiRes.i_wstart[j_u_HR][j_z_HR], i_wstart_min);
					i_wstride_max  = _MAX_(A_hiRes.i_wstride [j_u_HR][j_z_HR], i_wstride_max);
				}
			}

			A->i_wstride_max = _MAX_(i_wstride_max - i_wstart_min + 1, A->i_wstride_max);
			A->i_wstart[j_u][j_z] = i_wstart_min;
			A->i_wstride[j_u][j_z]  = i_wstride_max;
			}
	}

/*	printf("A->i_wstride_max = %d\n", A->i_wstride_max);
	printf("A->N_u = %d\n", A->N_u);
	printf("A->Delta_u = %lf\n", A->Delta_u);*/



	A->B = 			(float ****)	mem_alloc_4D(imgParams->N_x, imgParams->N_y,  sinoParams->N_beta, A->i_vstride_max, sizeof(float));
    A->j_u = 		(int ***)   	mem_alloc_3D(imgParams->N_x, imgParams->N_y,  sinoParams->N_beta, sizeof(int));
    




    /**
     * 		Merge colluns
     */
	for (j_x = 0; j_x < imgParams->N_x; ++j_x)
	{
		for (j_y = 0; j_y < imgParams->N_y; ++j_y)
		{
			for (i_beta = 0; i_beta < sinoParams->N_beta; ++i_beta)
			{
				/**
				 * 		Sum all subvoxel columns
				 */
				for (j_x_sub = 0; j_x_sub < hiResFactor; ++j_x_sub)
				{
					for (j_y_sub = 0; j_y_sub < hiResFactor; ++j_y_sub)
					{
						j_x_HR = j_x * hiResFactor + j_x_sub;
						j_y_HR = j_y * hiResFactor + j_y_sub;

						

						for (i_v = A_hiRes.i_vstart[j_x_HR][j_y_HR][i_beta]; i_v <= A_hiRes.i_vstride[j_x_HR][j_y_HR][i_beta]; ++i_v)
						{
							A->B[j_x][j_y][i_beta][i_v-A->i_vstart[j_x][j_y][i_beta]] += A_hiRes.B[j_x_HR][j_y_HR][i_beta][i_v-A_hiRes.i_vstart[j_x_HR][j_y_HR][i_beta]];
						}


					}

				}
/*				if (j_x == 2 && j_y == 2)
				{
					printf("B[%3d][%3d][%3d] = [", j_x, j_y, i_beta);
					for (i_v = A->i_vstart[j_x][j_y][i_beta]; i_v <= A->i_vstride[j_x][j_y][i_beta]; ++i_v)
					{
						printf("%8.6lf ", A->B[j_x][j_y][i_beta][i_v-A->i_vstart[j_x][j_y][i_beta]]);
					}
					printf("]\n");
					printf("i_v =              [");
					for (i_v = A->i_vstart[j_x][j_y][i_beta]; i_v <= A->i_vstride[j_x][j_y][i_beta]; ++i_v)
					{
						printf("%8d ", i_v);
					}
					printf("]\n\n");
				}*/
				A->j_u[j_x][j_y][i_beta] = ((A->u_v[j_x][j_y][i_beta] - imgParams->Delta_xy/2) - A->u_0) / A->Delta_u;
				if (A->j_u[j_x][j_y][i_beta] < 0 || A->j_u[j_x][j_y][i_beta] > A->N_u-1)
				{
					printf("WARNING j_u out of bounds!\n");
					printf("A->u_v[j_x][j_y][i_beta] = %lf\n", A->u_v[j_x][j_y][i_beta]);
					printf("A->j_u[j_x][j_y][i_beta] = %d\n\n", A->j_u[j_x][j_y][i_beta]);
				}

			}
		}
	}

	j_u_min = INT_MAX;
	j_u_max = INT_MIN;
	for (j_x = 0; j_x < imgParams->N_x; ++j_x)
	{
		for (j_y = 0; j_y < imgParams->N_y; ++j_y)
		{
			for (i_beta = 0; i_beta < sinoParams->N_beta; ++i_beta)
			{
				j_u_min = _MIN_(A->j_u[j_x][j_y][i_beta], j_u_min);
				j_u_max = _MAX_(A->j_u[j_x][j_y][i_beta], j_u_max);
			}
		}
	}


    A->C = 			(float ***) 	mem_alloc_3D(A->N_u, imgParams->N_z,  A->i_wstride_max, sizeof(float));



	for (j_u = 0; j_u < A->N_u; ++j_u)
	{
		for (j_z = 0; j_z < imgParams->N_z; ++j_z)
		{
			for (j_u_sub = 0; j_u_sub < hiResFactor; ++j_u_sub)
			{
				for (j_z_sub = 0; j_z_sub < hiResFactor; ++j_z_sub)
				{
					j_u_HR = j_u * hiResFactor + (int) round(j_u_sub * imgParams->Delta_xy / (A_hiRes.Delta_u * hiResFactor));
					j_z_HR = j_z * hiResFactor + j_z_sub;

					for (i_w = A_hiRes.i_wstart[j_u_HR][j_z_HR]; i_w <= A_hiRes.i_wstride[j_u_HR][j_z_HR]; ++i_w)
					{
						A->C[j_u][j_z][i_w-A->i_wstart[j_u][j_z]] += A_hiRes.C[j_u_HR][j_z_HR][i_w-A_hiRes.i_wstart[j_u_HR][j_z_HR]];
					}

				}
			}
		}
	}



}