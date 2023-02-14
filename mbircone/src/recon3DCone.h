#include "MBIRModularUtilities3D.h"
#include "icd3d.h"


void MBIR3DCone(struct Image *img, struct Sino *sino, struct ReconParams *reconParams, struct SysMatrix *A);

void MBIR3DDenoise(struct Image *img, struct Image *img_noisy, struct ReconParams *reconParams);
