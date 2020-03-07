#ifndef IO4D_H
#define IO4D_H


#include <getopt.h>
#include <stdio.h>
#include <string.h>
#include "stdlib.h"
#include "allocate.h"
#include "MBIRModularUtilities.h"

struct CmdLine{
    
    char masterFile[1000];
    char plainParamsFile[1000];
};


void readCmdLine(int argc, char *argv[], struct CmdLine *cmdLine);

void printCmdLine(struct CmdLine *cmdLine);

/**
 * 		General File IO
 */
void readLineFromFile(char *fName, int lineNr, char *str, int strSize);

void readStringFromFile(char *fName, char *str, int lineNr);

int readIntFromFile(char *fName, int lineNr);

double readDoubleFromFile(char *fName, int lineNr);

void absolutePath_relativeToFileLocation(char *fName_rel, char *base_file, char *fName_abs);

int str2int(char *str);

double str2double(char *str);


/**
 * 		Read/print text files
 */
void free_pathName(struct PathNames *pathNames);

void readBinaryFNames(char *masterFile, char *plainParamsFile, struct PathNames *pathNames);

void readImageParamsFromBinaryFile(struct PathNames *pathNames, struct ImageParams *imgParams);

void readParams(char *masterFile, char *plainParamsFile, struct Params *params);


void printPathNames(struct PathNames *pathNames);

void printImgParams(struct ImageParams *params);

void printParams(struct Params *params);


void printICDinfo(struct ICDInfo *icdInfo);

void printNeighborhoodInfo(struct NeighborhoodInfo *neighborhoodInfo);

void printNeighborsInImg(struct ICDInfo *icdInfo, struct Image *img);


int getDataTypeSize(char *dataType);

void read3DData(char *fName, void ***arr, int N1, int N2, int N3, char *dataType);

void write3DData(char *fName, void ***arr, int N1, int N2, int N3, char *dataType);


long int keepWritingToBinaryFile(FILE *fp, void *var, long int numEls, int elSize, char *fName);

long int keepReadingFromBinaryFile(FILE *fp, void *var, long int numEls, int elSize, char *fName);


void printFileIOInfo( char* message, char* fName, int size, char mode);

void printProgressOfLoop( int indexOfLoop, int NumIterations);



void logAndDisp_message(char *fName, char* message);

void log_message(char *fName, char* message);

void resetFile(char *fName);


#endif /* #ifndef IO4D_H */

