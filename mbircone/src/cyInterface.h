#ifndef _CY_INTERFACE_H_
#define _CY_INTERFACE_H_

int AmatrixComputeToFile(char *fName, double *angles, struct SinoParams *sinoParams, struct ImageParams *imgParams);

void allocate_3darray_from_flattened(struct matrix_float *mat, struct flat_array_2D *array);
void deallocate_3darray_to_flattened(struct matrix_float *mat);

#endif /* _CY_INTERFACE_H_ */