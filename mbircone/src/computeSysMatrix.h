

#include <stdio.h>
#include <math.h>
#include "allocate.h"
#include "MBIRModularUtilities3D.h"

void computeSysMatrix(struct SinoParams *sinoParams, struct ImageParams *imgParams, struct SysMatrix *A, struct ViewAngleList *viewAngleList);

void computeAMatrixParameters(struct SinoParams *sinoParams, struct ImageParams *imgParams, struct SysMatrix *A, struct ViewAngleList *viewAngleList);

void computeBMatrix(struct SinoParams *sinoParams, struct ImageParams *imgParams, struct SysMatrix *A, struct ViewAngleList *viewAngleList);

void computeCMatrix( struct SinoParams *sinoParams, struct ImageParams *imgParams, struct SysMatrix *A);
