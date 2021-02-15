#ifndef IO3D_H
#define IO3D_H


#include <getopt.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "allocate.h"
#include "MBIRModularUtilities3D.h"

struct CmdLine{
    
    char masterFile[1000];
    char plainParamsFile[1000];
    char modes[1000];
};


void readCmdLine(int argc, char *argv[], struct CmdLine *cmdLine);

void printCmdLine(struct CmdLine *cmdLine);

/**
 * 		General File IO
 */
void readLineFromFile(char *fName, char lineNr, char *str, int strSize);

void absolutePath_relativeToFileLocation(char *fName_rel, char *base_file, char *fName_abs);

long int str2int(char *str);

double str2double(char *str);


/**
 * 		Read/print text files
 */
void readBinaryFNames(char *masterFile, char *plainParamsFile, struct PathNames *pathNames);

void readSinoParams(char *masterFile, char *plainParamsFile, struct SinoParams *sinoParams);

void readImageParams(char *masterFile, char *plainParamsFile, struct ImageParams *imgParams);

void readReconParams(char *masterFile, char *plainParamsFile, struct ReconParams *reconParams);

void readAndAllocateViewAngleList(char *masterFile, char *plainParamsFile, struct ViewAngleList *list, struct SinoParams *sinoParams);


void printPathNames(struct PathNames *pathNames);

void printSinoParams(struct SinoParams *params);
/**/

void printImgParams(struct ImageParams *params);


void printReconParams(struct ReconParams *params);

void printSysMatrixParams(struct SysMatrix *A);

int getDataTypeSize(char *dataType);

void prependToFName(char *prefix, char *fName);

void read3DData(char *fName, void ***arr, long int N1, long int N2, long int N3, char *dataType);

void write3DData(char *fName, void ***arr, long int N1, long int N2, long int N3, char *dataType);



#endif /* #ifndef IO3D_H */

