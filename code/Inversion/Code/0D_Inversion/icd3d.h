
#ifndef ICD_H
#define ICD_H

#include "../0A_CLibraries/MBIRModularUtilities3D.h"


void ICDStep3DCone(struct Sino *sino, struct ImageF *img, struct SysMatrix *A, struct ICDInfo3DCone *icdInfo, struct ReconParams *reconParams, struct ReconAux *reconAux);

void prepareICDInfo(long int j_x, long int j_y, long int j_z, struct ICDInfo3DCone *icdInfo, struct ImageF *img, struct ReconAux *reconAux, struct ReconParams *reconParams);

void extractNeighbors( struct ICDInfo3DCone *icdInfo, struct ImageF *img, struct ReconParams *reconParams);

void computeTheta1Theta2ForwardTerm(struct Sino *sino, struct SysMatrix *A, struct ICDInfo3DCone *icdInfo, struct ReconParams *reconParams);

void computeTheta1Theta2PriorTermQGGMRF(struct ICDInfo3DCone *icdInfo, struct ReconParams *reconParams);

void computeTheta1Theta2PriorTermProxMap(struct ICDInfo3DCone *icdInfo, struct ReconParams *reconParams);

double surrogateCoeffQGGMRF(double Delta, struct ReconParams *reconParams);

void updateErrorSinogram(struct Sino *sino, struct SysMatrix *A, struct ICDInfo3DCone *icdInfo);

void updateIterationStats(struct ReconAux *reconAux, struct ICDInfo3DCone *icdInfo, struct ImageF *img);

void resetIterationStats(struct ReconAux *reconAux);


void RandomAux_ShuffleOrderXYZ(struct RandomAux *aux, struct ImageFParams *params);

void indexExtraction3D(long int j_xyz, long int *j_x, long int N_x, long int *j_y, long int N_y, long int *j_z, long int N_z);



double MAPCost3D(struct Sino *sino, struct ImageF *img, struct ReconParams *reconParams);

double MAPCostForward(struct Sino *sino);

double MAPCostPrior_QGGMRF(struct ImageF *img, struct ReconParams *reconParams);

double MAPCostPrior_ProxMap(struct ImageF *img, struct ReconParams *reconParams);

double MAPCostPrior_QGGMRFSingleVoxel_HalfNeighborhood(struct ICDInfo3DCone *icdInfo, struct ReconParams *reconParams);

double QGGMRFPotential(double delta, struct ReconParams *reconParams);


void partialZipline_computeStartStopIndex(long int *j_z_start, long int *j_z_stop, long int indexZiplines, long int numVoxelsPerZipline, long int N_z);

void prepareICDInfoRandGroup(long int j_x, long int j_y, struct RandomZiplineAux *randomZiplineAux, struct ICDInfo3DCone *icdInfo, struct ImageF *img, struct ReconParams *reconParams, struct ReconAux *reconAux);

void computeDeltaXjAndUpdate(struct ICDInfo3DCone *icdInfo, struct ReconParams *reconParams, struct ImageF *img, struct ReconAux *reconAux);

void computeDeltaXjAndUpdateGroup(struct ICDInfo3DCone *icdInfo, struct RandomZiplineAux *randomZiplineAux, struct ReconParams *reconParams, struct ImageF *img, struct ReconAux *reconAux);

void updateIterationStatsGroup(struct ReconAux *reconAux, struct ICDInfo3DCone *icdInfoArray, struct RandomZiplineAux *randomZiplineAux, struct ImageF *img, struct ReconParams *reconParams);

void dispAndLog_iterationInfo(struct ReconAux *reconAux, struct ReconParams *reconParams, int itNumber, int MaxIterations, double cost, double relUpdate, double stopThresholdChange, double weightScaler, double voxelsPerSecond, double ticToc_iteration, double weightedNormSquared_e, double ratioUpdated, double RRMSE, double stopThesholdRRMSE, double totalEquits);

double computeRelUpdate(struct ReconAux *reconAux, struct ReconParams *reconParams, struct ImageF *img);

void writeICDLoopStatus2File(char *fName, long int index, long int MaxIndex, int itNumber, double voxelsPerSecond);

/* * * * * * * * * * * * parallel * * * * * * * * * * * * **/
void prepareParallelAux(struct ParallelAux *parallelAux, long int N_M_max);

void freeParallelAux(struct ParallelAux *parallelAux);

void ICDStep3DConeGroup(struct Sino *sino, struct ImageF *img, struct SysMatrix *A, struct ICDInfo3DCone *icdInfo, struct ReconParams *reconParams, struct RandomZiplineAux *randomZiplineAux, struct ParallelAux *parallelAux, struct ReconAux *reconAux);

void computeTheta1Theta2ForwardTermGroup(struct Sino *sino, struct SysMatrix *A, struct ICDInfo3DCone *icdInfo, struct RandomZiplineAux *randomZiplineAux, struct ParallelAux *parallelAux, struct ReconParams *reconParams);

void computeTheta1Theta2PriorTermQGGMRFGroup(struct ICDInfo3DCone *icdInfo, struct ReconParams *reconParams, struct RandomZiplineAux *randomZiplineAux);

void updateErrorSinogramGroup(struct Sino *sino, struct SysMatrix *A, struct ICDInfo3DCone *icdInfo, struct RandomZiplineAux *randomZiplineAux);

void computeTheta1Theta2PriorTermProxMapGroup(struct ICDInfo3DCone *icdInfo, struct ReconParams *reconParams, struct RandomZiplineAux *randomZiplineAux);

/* * * * * * * * * * * * time aux ICD * * * * * * * * * * * * **/

void speedAuxICD_reset(struct SpeedAuxICD *speedAuxICD);

void speedAuxICD_update(struct SpeedAuxICD *speedAuxICD, long int incrementNumber);

void speedAuxICD_computeSpeed(struct SpeedAuxICD *speedAuxICD);

/* * * * * * * * * * * * NHICD * * * * * * * * * * * * **/

int NHICD_isVoxelHot(struct ReconParams *reconParams, struct ImageF *img, long int j_x, long int j_y, long int j_z, double lastChangeThreshold);

int NHICD_activatePartialUpdate(struct ReconParams *reconParams, double relativeWeightedForwardError);

int NHICD_checkPartialZiplineHot(struct ReconAux *reconAux, long int j_x, long int j_y, long int indexZiplines, struct ImageF *img);

void NHICD_checkPartialZiplinesHot(struct ReconAux *reconAux, long int j_x, long int j_y, struct ReconParams *reconParams, struct ImageF *img);

void updateNHICDStats(struct ReconAux *reconAux, long int j_x, long int j_y, struct ImageF *img, struct ReconParams *reconParams);

#endif

