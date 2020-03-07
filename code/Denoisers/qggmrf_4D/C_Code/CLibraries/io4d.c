
#include "io4d.h"
#include <libgen.h>
#include "../../../../plainParams/plainParams.h"



/* Read Command-line */
void readCmdLine(int argc, char *argv[], struct CmdLine *cmdLine)
{
    char ch;

    

    cmdLine->masterFile[0] = '\0';
    cmdLine->masterFile[0] = '\0';

    /* get options */
    while ((ch = getopt(argc, argv, "a:b:")) != EOF)
    {

        switch (ch)
        {
            case 'a':
            {
                sprintf(cmdLine->masterFile, "%s", optarg);
                break;
            }
            case 'b':
            {
                sprintf(cmdLine->plainParamsFile, "%s", optarg);
                break;
            }           
            default:
            {
                printf("\nError in readCmdLine: Argument \"-%c\" not recognized\n", ch);
                exit(-1);
                break;
            }
        }
    }
    if(cmdLine->masterFile[0] == '\0')
    {
        printf("\nError in readCmdLine: masterFile not specified\n");
        exit(-1);
    }
    if(cmdLine->plainParamsFile[0] == '\0')
    {
        printf("\nError in readCmdLine: plainParamsFile not specified\n");
        exit(-1);
    }
}

void printCmdLine(struct CmdLine *cmdLine)
{
    char str[500];
    sprintf(str,    "\n"
                    "Command line arguments read:\n"
                    "\tmasterFile file = %s\n"
                    "\tplainParamsFile file = %s\n", cmdLine->masterFile, cmdLine->plainParamsFile);                    
    logAndDisp_message(LOG_PROGRESS, str);
}

void readLineFromFile(char *fName, int lineNr, char *line, int lineSize)
{
    /**
     *      Returns the specified line. Result is in "line". Newline character removed.
     */
    int i, reachedEOF;
    FILE *fid;


    fid = fopen(fName, "r");
    if (fid == NULL)
    {
        fprintf(stderr, "ERROR in readLineFromFile: Failed to open file \"%s\"", line);
        exit(-1);
    }

    reachedEOF = 0;
    for(i = 0; i < lineNr ; ++i)
    {
        if(0 == fgets(line, lineSize, fid))
        {
            reachedEOF = 1;
            break;
        }
    }
    if(reachedEOF)
    {
        fprintf(stderr, "ERROR in readLineFromFile: File \"%s\" has less than %d lines\n", fName, lineNr);
        exit(-1);
    }

    /* Remove newline character at the end of the line*/
    line[strlen(line)-1] = '\0';

    fclose(fid);

}

void readStringFromFile(char *fName, char *str, int lineNr)
{
    char line[500];
    
    readLineFromFile(fName, lineNr, line, sizeof(line));
    strcpy(str, line);
}

int readIntFromFile(char *fName, int lineNr)
{
    char line[500];
    int intNumber;

    readLineFromFile(fName, lineNr, line, sizeof(line));
    sscanf(line, "%d", &intNumber);

    return intNumber;
}

double readDoubleFromFile(char *fName, int lineNr)
{
    char line[500];
    double doubleNumber;

    readLineFromFile(fName, lineNr, line, sizeof(line));
    sscanf(line, "%lf", &doubleNumber);

    return doubleNumber;
}

void absolutePath_relativeToFileLocation(char *fName_rel, char *base_file, char *fName_abs)
{
    char directory_backup[1000];
    char *dir_of_base_file;
    char buff[1000];

    getcwd(directory_backup, 1000);

    strcpy(buff, base_file); /* Protect base_file from being changed */
    dir_of_base_file = dirname(buff);
    
    chdir(dir_of_base_file);
    realpath(fName_rel, fName_abs);
    chdir(directory_backup);

}

int str2int(char *str)
{
    int value;
    sscanf(str, "%d", &value);
    return value;
}

double str2double(char *str)
{
    double value;
    sscanf(str, "%lf", &value);
    return value;
}


void free_pathName(struct PathNames *pathNames)
{
    mem_free_2D( (void **)(pathNames->noisyImageNames) );
    mem_free_2D( (void **)(pathNames->denoisedImageNames) );
}


void readBinaryFNames(char *masterFile, char *plainParamsFile, struct PathNames *pathNames)
{
    char get_set[1000] = "get";
    char resolveFlag[100] = "-r";
    int i;
    char str[STRLEN];

    /* Read files containing list of filenames */
    plainParams(plainParamsFile, get_set, masterFile, "noisyBinaryFName_timeList", "", pathNames->noisyBinaryFName_timeList, resolveFlag);
    plainParams(plainParamsFile, get_set, masterFile, "denoisedBinaryFName_timeList", "", pathNames->denoisedBinaryFName_timeList, resolveFlag);

    pathNames->N_t = readIntFromFile(pathNames->noisyBinaryFName_timeList, 1);

    pathNames->noisyImageNames = (char**)mem_alloc_2D( pathNames->N_t, STRLEN, sizeof(char) );
    pathNames->denoisedImageNames = (char**)mem_alloc_2D( pathNames->N_t, STRLEN, sizeof(char) );

    /*printf("Num time paths = %d\n",pathNames->N_t );*/

    for(i=0; i< pathNames->N_t; i++){

        readStringFromFile(pathNames->noisyBinaryFName_timeList, str, i+2);
        absolutePath_relativeToFileLocation( str, pathNames->noisyBinaryFName_timeList, pathNames->noisyImageNames[i] );
        /*printf("Noisy path = %s\n",pathNames->noisyImageNames[i] );*/

        readStringFromFile(pathNames->denoisedBinaryFName_timeList, str, i+2);
        absolutePath_relativeToFileLocation( str, pathNames->denoisedBinaryFName_timeList, pathNames->denoisedImageNames[i] );
        /*printf("Denoised path = %s\n",pathNames->denoisedImageNames[i] );*/
    }
}

void readImageParamsFromBinaryFile(struct PathNames *pathNames, struct ImageParams *imgParams)
{
    FILE* fp;
    int buff[3];
    int N1_is, N2_is, N3_is;
    int numToRead;
    int numHaveRead;

    fp = fopen (pathNames->noisyImageNames[0], "r" );
    if(fp == NULL)
    {
        printf("readImageParamsFromBinaryFile: Error in opening file \"%s\".\n", pathNames->noisyImageNames[0]);
        goto error;
    }

    /**
     *      Read dimesions in header
     */
    numToRead = 3;
    numHaveRead = fread(&buff, sizeof(int), numToRead, fp);
    if (numHaveRead != numToRead)
    {
        printf("Error in reading header in file \"%s\".\n", pathNames->noisyImageNames[0]);
        printf("Number of elements read does not match required\n");
        printf("numToRead = %d, numHaveRead = %d\n", numToRead, numHaveRead);
        goto error;
    }

    /* Get image sizes from header */
    N1_is = buff[0];
    N2_is = buff[1];
    N3_is = buff[2];
    
    imgParams->N_x = N1_is;
    imgParams->N_y = N2_is;
    imgParams->N_z = N3_is;
    imgParams->N_t = pathNames->N_t;

    /* if updated, also update "printImgParams" */

    fclose(fp);
    return;

error:
    if (fp)
        fclose(fp);
    exit(-1);

}

void readParams(char *masterFile, char *plainParamsFile, struct Params *params)
{
    char get_set[1000] = "get";
    char masterField[1000] = "params";
    char temp[1000] = "";
    char resolveFlag[100] = "";

    plainParams(plainParamsFile, get_set, masterFile, masterField, "is_positivity_constraint", temp, resolveFlag);
    params->is_positivity_constraint = str2int(temp);

    plainParams(plainParamsFile, get_set, masterFile, masterField, "q", temp, resolveFlag);
    params->q = str2double(temp);

    plainParams(plainParamsFile, get_set, masterFile, masterField, "p", temp, resolveFlag);
    params->p = str2double(temp);


    plainParams(plainParamsFile, get_set, masterFile, masterField, "T_s", temp, resolveFlag);
    params->T_s = str2double(temp);

    plainParams(plainParamsFile, get_set, masterFile, masterField, "T_t", temp, resolveFlag);
    params->T_t = str2double(temp);

    plainParams(plainParamsFile, get_set, masterFile, masterField, "sigma_s", temp, resolveFlag);
    params->sigma_s = str2double(temp);

    plainParams(plainParamsFile, get_set, masterFile, masterField, "sigma_t", temp, resolveFlag);
    params->sigma_t = str2double(temp);

    plainParams(plainParamsFile, get_set, masterFile, masterField, "sigma", temp, resolveFlag);
    params->sigma = str2double(temp);


    plainParams(plainParamsFile, get_set, masterFile, masterField, "spacePriorMode", temp, resolveFlag);
    params->spacePriorMode = str2int(temp);
    
    plainParams(plainParamsFile, get_set, masterFile, masterField, "isTimePrior", temp, resolveFlag);
    params->isTimePrior = str2int(temp);

    
    plainParams(plainParamsFile, get_set, masterFile, masterField, "stopThreshold", temp, resolveFlag);
    params->stopThreshold = str2double(temp);

    plainParams(plainParamsFile, get_set, masterFile, masterField, "MaxIterations", temp, resolveFlag);
    params->MaxIterations = str2int(temp);


    plainParams(plainParamsFile, get_set, masterFile, masterField, "isSaveImage", temp, resolveFlag);
    params->isSaveImage = str2int(temp);

    plainParams(plainParamsFile, get_set, masterFile, masterField, "verbose", temp, resolveFlag);
    params->verbose = str2int(temp);
    

    /* if updated, also update "printParams" */

}


void printPathNames(struct PathNames *pathNames)
{
    char str[10000];
    int i;

    sprintf(str, "\nPath Names Read:\n");

    sprintf(str, "%s    masterFile = %s \n", str, pathNames->masterFile);
    sprintf(str, "%s    plainParamsFile = %s \n", str, pathNames->plainParamsFile);
    sprintf(str, "%s\n", str);
    sprintf(str, "%s    noisyBinaryFName_timeList = %s \n", str, pathNames->noisyBinaryFName_timeList);
    sprintf(str, "%s    denoisedBinaryFName_timeList = %s \n", str, pathNames->denoisedBinaryFName_timeList);
    sprintf(str, "%s    Num files = %d \n", str, pathNames->N_t);
    sprintf(str, "%s\n", str);
    logAndDisp_message(LOG_PROGRESS, str);

/*  sprintf(str, "\nBinary path Names:\n");
    for(i=0; i< pathNames->N_t; i++){
        sprintf(str, "%s\n t=%d \n", str, i);
        sprintf(str, "%s noisyImageName = %s \n", str, pathNames->noisyImageNames[i]);
        sprintf(str, "%s denoisedImageName = %s \n", str, pathNames->denoisedImageNames[i]);    
    }

    logAndDisp_message(LOG_PROGRESS, str);*/
}

void printImgParams(struct ImageParams *params)
{
    char str[2000];
    sprintf(str, "\nImage parameters read:\n");

    sprintf(str, "%s\tN_t = %d \n", str, params->N_t);
    sprintf(str, "%s\tN_x = %d \n", str, params->N_x);
    sprintf(str, "%s\tN_y = %d \n", str, params->N_y);
    sprintf(str, "%s\tN_z = %d \n", str, params->N_z);
    sprintf(str, "%s\n", str);

    logAndDisp_message(LOG_PROGRESS, str);

}


void printParams(struct Params *params)
{
    char str[2000];

    sprintf(str, "\nReconstruction parameters read:\n");

    sprintf(str, "%s\tis_positivity_constraint = %d \n", str, params->is_positivity_constraint);

    sprintf(str, "%s\tq = %e \n", str, params->q);
    sprintf(str, "%s\tp = %e \n", str, params->p);

    sprintf(str, "%s\tT_s = %e \n", str, params->T_s);
    sprintf(str, "%s\tT_t = %e \n", str, params->T_t);
    sprintf(str, "%s\tsigma_s = %e \n", str, params->sigma_s);
    sprintf(str, "%s\tsigma_t = %e \n", str, params->sigma_t);

    sprintf(str, "%s\tsigma = %e \n", str, params->sigma);

    sprintf(str, "%s\tspacePriorMode = %d \n", str, params->spacePriorMode);
    sprintf(str, "%s\tisTimePrior = %d \n", str, params->isTimePrior);

    sprintf(str, "%s\tstopThreshold = %e \n", str, params->stopThreshold);
    sprintf(str, "%s\tMaxIterations = %d \n", str, params->MaxIterations);

    sprintf(str, "%s\tisSaveImage = %d \n", str, params->isSaveImage);

    sprintf(str, "%s\tverbose = %d \n", str, params->verbose);

    sprintf(str, "%s\n", str);

    logAndDisp_message(LOG_PROGRESS, str);
}

void printICDinfo(struct ICDInfo *icdInfo)
{
    char str[2000], str2[2000];
    int i;

    sprintf(str, "\nICD Info:\n");

    sprintf(str, "%s\t j_t = %d \n", str, icdInfo->j_t);
    sprintf(str, "%s\t j_x = %d \n", str, icdInfo->j_x);
    sprintf(str, "%s\t j_y = %d \n", str, icdInfo->j_y);
    sprintf(str, "%s\t j_z = %d \n", str, icdInfo->j_z);

    sprintf(str, "%s\t Delta_xj = %e \n", str, icdInfo->Delta_xj);
    sprintf(str, "%s\t old_xj = %e \n", str, icdInfo->old_xj);

    sprintf(str, "%s\t theta1 = %e \n", str, icdInfo->theta1);
    sprintf(str, "%s\t theta2 = %e \n", str, icdInfo->theta2);

    sprintf(str, "%s\t theta1_F = %e \n", str, icdInfo->theta1_F);
    sprintf(str, "%s\t theta2_F = %e \n", str, icdInfo->theta2_F);
    sprintf(str, "%s\t theta1_P = %e \n", str, icdInfo->theta1_P);
    sprintf(str, "%s\t theta2_P = %e \n", str, icdInfo->theta2_P);
    sprintf(str, "%s\t %s\n", str, str2);

    sprintf(str, "%s\n", str);

    logAndDisp_message(LOG_PROGRESS, str);
    printNeighborhoodInfo( icdInfo->neighborhoodInfo );

}

void printNeighborhoodInfo(struct NeighborhoodInfo *neighborhoodInfo)
{
    char str[2000];
    int j_t, j_x, j_y, j_z;
    int neighborID;
    double wt_val;

    sprintf(str, "Neighborhood Info:\n");

    sprintf(str, "%s\t numNeighbors_t = %d  \n", str, neighborhoodInfo->numNeighbors_t );
    sprintf(str, "%s\t numNeighbors_s = %d  \n", str, neighborhoodInfo->numNeighbors_s );
    sprintf(str, "%s\t numNeighbors = %d \n", str, neighborhoodInfo->numNeighbors);
    sprintf(str, "%s\n", str);

    sprintf(str, "%sWts:\n\t", str);

    for(neighborID=0; neighborID<neighborhoodInfo->numNeighbors; neighborID++){

        j_t = neighborhoodInfo->j_t_arr[neighborID] ;
        j_x = neighborhoodInfo->j_x_arr[neighborID] ;
        j_y = neighborhoodInfo->j_y_arr[neighborID] ;
        j_z = neighborhoodInfo->j_z_arr[neighborID] ;

        wt_val = neighborhoodInfo->neighborWts[neighborID];
        sprintf(str, "%s\nt=%d x=%d y=%d z=%d. neighborWt=%f \n", str, j_t, j_x, j_y, j_z, wt_val );   
        
    }

    sprintf(str, "%s\n", str);

    logAndDisp_message(LOG_PROGRESS, str);
}

void printNeighborsInImg(struct ICDInfo *icdInfo, struct Image *img)
{
    char str[2000];
    int j_t, j_x, j_y, j_z;
    int neighbor_j_t, neighbor_j_x, neighbor_j_y, neighbor_j_z;
    int neighborID;
    double neighborVal;


    sprintf(str, "Neighbor List:\n\t");
    for(neighborID=0; neighborID < icdInfo->neighborhoodInfo->numNeighbors; neighborID++){

        j_t = icdInfo->neighborhoodInfo->j_t_arr[neighborID] ;
        j_x = icdInfo->neighborhoodInfo->j_x_arr[neighborID] ;
        j_y = icdInfo->neighborhoodInfo->j_y_arr[neighborID] ;
        j_z = icdInfo->neighborhoodInfo->j_z_arr[neighborID] ;


        neighbor_j_t = j_t + icdInfo->j_t;
        neighbor_j_x = j_x + icdInfo->j_x;
        neighbor_j_y = j_y + icdInfo->j_y;
        neighbor_j_z = j_z + icdInfo->j_z;

        if( isWithinVol( neighbor_j_t, neighbor_j_x, neighbor_j_y, neighbor_j_z, img) ){
            
            neighborVal = img->denoised[neighbor_j_t][neighbor_j_x][neighbor_j_y][neighbor_j_z];
            sprintf(str, "%s %f", str, neighborVal);
        }
        else{
            sprintf(str, "%s %f", str, 0.0);
        }


    }

    sprintf(str, "%s\n", str);

    logAndDisp_message(LOG_PROGRESS, str);
}


int getDataTypeSize(char *dataType)
{
    if (strcmp(dataType, "char") == 0) 
    {
        return sizeof(char);
    } 
    else if (strcmp(dataType, "float") == 0)
    {
        return sizeof(float);
    }
    else if (strcmp(dataType, "unsigned char") == 0)
    {
        return sizeof(unsigned char);
    }   else /* default: */
    {
        printf("ERROR: Error in getDataTypeSize: data type \"%s\" not recognized\n", dataType);
        exit(-1);
    }
}

void read3DData(char *fName, void ***arr, int N1, int N2, int N3, char *dataType)
{
    FILE* fp;
    int buff[3];
    int N1_is, N2_is, N3_is;
    int numToRead;
    int numHaveRead;

    int dataTypeSize;

    dataTypeSize = getDataTypeSize(dataType);

    fp = fopen (fName, "r" );
    if(fp == NULL)
    {
        printf("read3DData: Error in opening file \"%s\".\n", fName);
        goto error;
    }

    /**
     *      Read dimesions in header
     */
    numToRead = 3;
    numHaveRead = fread(&buff, sizeof(int), numToRead, fp);
    if (numHaveRead != numToRead)
    {
        printf("Error in reading header in file \"%s\".\n", fName);
        printf("Number of elements read does not match required\n");
        printf("numToRead = %d, numHaveRead = %d\n", numToRead, numHaveRead);
        goto error;
    }

    /**
     *      Check if header dimensions match the input dimensions
     */
    N1_is = buff[0];
    N2_is = buff[1];
    N3_is = buff[2];
    if (N1_is != N1 || N2_is != N2 || N3_is != N3)
    {
        printf("Error in reading file \"%s\".\n", fName);
        printf("Dimensions don't match\n");
        printf("Dimensions should: (%3d, %3d, %3d)\n", N1, N2, N3);
        printf("Dimensions   is  : (%3d, %3d, %3d)\n", N1_is, N2_is, N3_is);
        goto error;
    }

    /**
     *      Read body of data
     */
    numToRead = N1*N2*N3;
    numHaveRead = fread(&arr[0][0][0], dataTypeSize, numToRead, fp);
    if (numHaveRead != numToRead)
    {
        printf("Error in reading body file \"%s\".\n", fName);
        printf("Number of elements read does not match required\n");
        printf("numToRead = %d, numHaveRead = %d\n", numToRead, numHaveRead);
        goto error;
    }

    fclose(fp);
    return;


error:
    if (fp)
        fclose(fp);
    exit(-1);
}

void write3DData(char *fName, void ***arr, int N1, int N2, int N3, char *dataType)
{
    FILE* fp;
    int buff[3];
    int numToWrite;
    int numWritten;

    int dataTypeSize;

    dataTypeSize = getDataTypeSize(dataType);

    fp = fopen (fName, "w" );
    if(fp == NULL)
    {
        printf("Error in opening file \"%s\".\n", fName);
        goto error;
    }

    /**
     *      Write dimesions in header
     */
    buff[0] = N1;
    buff[1] = N2;
    buff[2] = N3;

    numToWrite = 3;
    numWritten = fwrite(&buff, sizeof(int), numToWrite, fp);
    if (numWritten != numToWrite)
    {
        printf("Error in writing header in file \"%s\".\n", fName);
        printf("Number of elements written does not match required\n");
        printf("numToWrite = %d, numWritten = %d\n", numToWrite, numWritten);
        goto error;
    }

    /**
     *      Write body of data
     */
    numToWrite = N1*N2*N3;
    numWritten = fwrite(&arr[0][0][0], dataTypeSize, numToWrite, fp);
    if (numWritten != numToWrite)
    {
        printf("Error in reading body file \"%s\".\n", fName);
        printf("Number of elements written does not match required\n");
        printf("numToWrite = %d, numWritten = %d\n", numToWrite, numWritten);
        goto error;
    }

    fclose(fp);
    return;


error:
    if (fp)
        fclose(fp);
    exit(-1);
}



long int keepWritingToBinaryFile(FILE *fp, void *var, long int numEls, int elSize, char *fName)
{
    /* Return number of bytes written */
    long int numElsWritten;

    numElsWritten = fwrite(var, elSize, numEls, fp);
    if(numElsWritten != numEls)
    {
        fprintf(stderr, "ERROR in keepWritingToBinaryFile: file \"%s\" terminated early.\n", fName);
        fprintf(stderr, "Tried to write %li elements of size %d Bytes. Wrote %li elements.\n", numEls, elSize, numElsWritten);

        fclose(fp);
        exit(-1);
    }
    return (long int) numEls * elSize;
}

long int keepReadingFromBinaryFile(FILE *fp, void *var, long int numEls, int elSize, char *fName)
{
    /* Return number of bytes read */
    long int numElsRead;

    numElsRead = fread(var, elSize, numEls, fp);
    if(numElsRead != numEls)
    {
        fprintf(stderr, "ERROR in keepReadingFromBinaryFile: file \"%s\" terminated early.\n", fName);
        fprintf(stderr, "Tried to read %li elements of size %d Bytes. Read %li elements.\n", numEls, elSize, numElsRead);
        fclose(fp);
        exit(-1);
    }
    return (long int) numEls * elSize;
}


void printFileIOInfo( char* functionName, char* fName, int size, char mode)
{
    char readwrite[200];    /* puts the word "Read" or "Write" into the output */
    switch(mode)
    {
        case 'r':   strcpy(readwrite, "Read "); break;
        case 'w':   strcpy(readwrite, "Write"); break;
        default:    printf("Error in printFileIOInfo: Use mode 'r' or 'w'\n");
                    exit(-1);
    }
    printf("\n");
    printf("    ************** FILE ACCESS ********************************\n");
    printf(" ****  File access in: %s\n", functionName);
    printf("*****  File name     : %s\n", fName);
    printf("*****  %-14s: %-15d bytes\n", readwrite, size);
    printf("*****                = %-15e kB\n", (double) size*1e-3);
    printf(" ****                = %-15e MB\n", (double) size*1e-6);
    printf("    ***********************************************************\n");
}

void printProgressOfLoop( int indexOfLoop, int NumIterations)
{
    double percent;

    percent = (double) (1+indexOfLoop) / (double) NumIterations * 100;
    printf("\r[%.1e%%]", percent );
    fflush(stdout); 

}

void logAndDisp_message(char *fName, char* message)
{
    log_message(fName, message);
    printf("%s", message);
}

void log_message(char *fName, char* message)
{
    FILE *fp;

    fp = fopen(fName, "a");
    if (fp != NULL)
    {
        fprintf(fp, "%s", message);
        fclose(fp);
    }
    else
    {
        fprintf(stderr, "WARNING: In log_message: Could not open file %s\n", fName);
    }

}


void resetFile(char *fName)
{
    FILE *filePointer;

    filePointer = fopen(fName, "w");
    fclose(filePointer);
        
}







