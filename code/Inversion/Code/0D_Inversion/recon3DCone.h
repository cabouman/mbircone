#include "../0A_CLibraries/MBIRModularUtilities3D.h"
#include "../0A_CLibraries/io3d.h"
#include "icd3d.h"



void MBIR3DCone(struct ImageF *img, struct Sino *sino, struct ReconParams *reconParams, struct SysMatrix *A, struct PathNames *pathNames);

void MBIR3DCone_OMP(struct ImageF *img, struct Sino *sino, struct ReconParams *reconParams, struct SysMatrix *A, struct PathNames *pathNames);

