
#ifndef ICD_H
#define ICD_H

#include "../CLibraries/MBIRModularUtilities.h"


void ICDStep3DCone(struct Image *img, struct ICDInfo *icdInfo, struct Params *params);

/* Neighborhood and icd structures  */

void initialize_NeighborhoodInfo(struct NeighborhoodInfo *neighborhoodInfo, struct Image *img, struct Params *params);

int findNumNeighbors_t(struct Params *params);
int findNumNeighbors_s(struct Params *params);

int isNeighbor(int j_t, int j_x, int j_y, int j_z, struct Params *params);

void free_NeighborhoodInfo(struct NeighborhoodInfo *neighborhoodInfo);

void prepareICDInfo(int j_t, int j_x, int j_y, int j_z, struct ICDInfo *icdInfo, struct NeighborhoodInfo *neighborhoodInfo, struct Image *img);


/* Compute thetas  */

void computeTheta1Theta2ForwardTerm(struct Image *img, struct ICDInfo *icdInfo, struct Params *params);

void computeTheta1Theta2PriorTermQGGMRF(struct ICDInfo *icdInfo, struct Params *params, struct Image *img);

void computeTheta1Theta2(struct ICDInfo *icdInfo);

double surrogateCoeffQGGMRF(double Delta, double p, double q, double T, double sigmaX);



void updateIterationStats(double *TotalValueChange, double *TotalVoxelValue, int *NumUpdatedVoxels, struct ICDInfo *icdInfo, struct Image *img);

void resetIterationStats(double *TotalValueChange, double *TotalVoxelValue, int *NumUpdatedVoxels);

void RandomAux_ShuffleorderTXYZ(struct RandomAux *aux, struct ImageParams *params);

void indexExtraction4D(int j_txyz, int *j_t, int N_t, int *j_x, int N_x, int *j_y, int N_y, int *j_z, int N_z);


/* Compute costs  */

double MAPCost4D(struct Image *img, struct Params *params, struct NeighborhoodInfo *neighborhoodInfo);

double MAPCostForward(struct Image *img, struct Params *params);

double MAPCostPrior_QGGMRF(struct Image *img, struct Params *params, struct NeighborhoodInfo *neighborhoodInfo);

double MAPCostPrior_QGGMRFSingleVoxel_HalfNeighborhood(struct Image *img, struct ICDInfo *icdInfo, struct Params *params);

double QGGMRFPotential(double delta, double p, double q, double T, double sigmaX);


/* Updates */

void computeDeltaXjAndUpdate(struct ICDInfo *icdInfo, struct Params *params, struct Image *img);

void dispAndLog_iterationInfo(int itNumber, int MaxIterations, double cost, double relUpdatePercent, double voxelsPerSecond, double ticToc_iteration, double normError);

double computeRelUpdate(int NumUpdatedVoxels, double TotalValueChange, double TotalVoxelValue);

void writeICDLoopStatus2File(char *fName, int index, int MaxIndex, int itNumber, double voxelsPerSecond);


/* * * * * * * * * * * * time aux ICD * * * * * * * * * * * * **/

void speedAuxICD_reset(struct SpeedAuxICD *speedAuxICD);

void speedAuxICD_update(struct SpeedAuxICD *speedAuxICD, int incrementNumber);

void speedAuxICD_computeSpeed(struct SpeedAuxICD *speedAuxICD);


#endif

