

#include <stdio.h>
#include <math.h>
#include "../0A_CLibraries/allocate.h"
#include "../0A_CLibraries/io3d.h"
#include "../0A_CLibraries/MBIRModularUtilities3D.h"

void computeSysMatrix(struct SinoParams *sinoParams, struct ImageFParams *imgParams, struct SysMatrix *A, struct ReconParams *reconParams, struct ViewAngleList *viewAngleList);

void computeAMatrixParameters(struct SinoParams *sinoParams, struct ImageFParams *imgParams, struct SysMatrix *A, struct ReconParams *reconParams, struct ViewAngleList *viewAngleList);

void computeBMatrix(struct SinoParams *sinoParams, struct ImageFParams *imgParams, struct SysMatrix *A, struct ReconParams *reconParams, struct ViewAngleList *viewAngleList);

void computeCMatrix( struct SinoParams *sinoParams, struct ImageFParams *imgParams, struct SysMatrix *A, struct ReconParams *reconParams);
