

#include <stdio.h>
#include <math.h>
#include "allocate.h"
#include "MBIRModularUtilities3D.h"

void computeSysMatrix(struct SinoParams *sinoParams, struct ImageParams *imgParams, struct SysMatrix *A, struct ViewAngleList *viewAngleList);

void computeAMatrixParameters(struct SinoParams *sinoParams, struct ImageParams *imgParams, struct SysMatrix *A, struct ViewAngleList *viewAngleList);

void computeBMatrix(struct SinoParams *sinoParams, struct ImageParams *imgParams, struct SysMatrix *A, struct ViewAngleList *viewAngleList);

void computeCMatrix( struct SinoParams *sinoParams, struct ImageParams *imgParams, struct SysMatrix *A);


void writeSysMatrix(char *fName, struct SinoParams *sinoParams, struct ImageParams *imgParams, struct SysMatrix *A);

void readSysMatrix(char *fName, struct SinoParams *sinoParams, struct ImageParams *imgParams, struct SysMatrix *A);


void allocateSysMatrix(struct SysMatrix *A, long int N_x, long int N_y, long int N_z, long int N_beta, long int i_vstride_max, long int i_wstride_max, long int N_u);

void freeSysMatrix(struct SysMatrix *A);