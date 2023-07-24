
#ifndef ICD_Denoise_H
#define ICD_Denoise_H

#include "MBIRModularUtilities3D.h"

void ICDStep3DDenoise(struct Image *img, struct Image *err_image, struct ICDInfo3DCone *icdInfo, struct ReconParams *reconParams);

void computeTheta1Theta2ForwardTermDenoise(struct Image *err_image, struct ICDInfo3DCone *icdInfo, struct ReconParams *reconParams);

float MAPCost3DDenoise(struct Image *image, struct Image *err_image, struct ReconParams *reconParams);

float MAPCostForwardDenoise(struct Image *err_image, struct ReconParams *reconParams);

void disp_iterationInfo_denoise(int itNumber, int MaxIterations, float cost, float relUpdate, float stopThresholdChange, float weightScaler_value, double ticToc_iteration);
#endif

