#include "io3d.h"
#include "MBIRModularUtilities3D.h"
#include "recon3DCone.h"
#include "computeSysMatrix.h"




int main(int argc, char *argv[])
{
    struct CmdLine cmdLine;
    struct PathNames pathNames;
    struct Sino sino;
    struct Image img;
    struct ReconParams reconParams;
    struct SysMatrix A;
    struct ViewAngleList viewAngleList;


    char delim[] = ",";
    char *mode;

    char isMode_sys = 0;
    char isMode_proj = 0;
    char isMode_wghtRecon = 0;
    char isMode_init = 0;
    char isMode_recon = 0;
    char isMode_backprojlike = 0;

    char isNeed_SysMatrix = 0;
    char isExistSysMatrix = 0;

    char isNeed_sino_DOT_vox = 0;
    char exist_sino_DOT_vox = 0;

    char isNeed_sino_DOT_wgt = 0;
    char exist_sino_DOT_wgt = 0;
    /**
     *      Reset Log files
     */
    resetFile(LOG_TIME);
    resetFile(LOG_PROGRESS);

    /**
     *      Process Command Line argument(s)
     */
    readCmdLine(argc, argv, &cmdLine);
    printCmdLine(&cmdLine);
    

    /**
     *      Loop through all modes in order of appearance
     */
    mode = strtok(cmdLine.modes, delim);
    while(mode != NULL)
    {


        if(strcmp(mode,"sys")==0)
        {
            isMode_sys = 1;
        }
        else if(strcmp(mode,"proj")==0)
        {
            isMode_proj = 1;
            isNeed_SysMatrix = 1;
        }
        else if(strcmp(mode,"backprojlike")==0)
        {
            isMode_backprojlike = 1;
            isNeed_SysMatrix = 1;
        }
        else if(strcmp(mode,"wghtRecon")==0)
        {
            isMode_wghtRecon = 1;
            isNeed_SysMatrix = 1;
            isNeed_sino_DOT_wgt = 1;
        }
        else if(strcmp(mode,"init")==0)
        {
            isMode_init = 1;
            isNeed_SysMatrix = 1;
            isNeed_sino_DOT_vox = 1;
            isNeed_sino_DOT_wgt = 1;
        }
        else if(strcmp(mode,"recon")==0)
        {
            isMode_recon = 1;
            isNeed_SysMatrix = 1;
            isNeed_sino_DOT_vox = 1;
            isNeed_sino_DOT_wgt = 1;
        }
        else
        {
            printf("ERROR: Mode '%s' unknown\n", mode);
            printf("Modes must be separated by ',': \n");
            printf("               sys\n");
            printf("               init\n");
            printf("               recon\n");
            printf("               proj\n");
            exit(-1);
        }
        mode = strtok(NULL, delim);
    }

    

    /**
     *      Read input text files
     */
    strcpy(pathNames.masterFile, cmdLine.masterFile);
    strcpy(pathNames.plainParamsFile, cmdLine.plainParamsFile);

    readImageParams(pathNames.masterFile, pathNames.plainParamsFile, &img.params);

    readReconParams(pathNames.masterFile, pathNames.plainParamsFile, &reconParams);
    computeSecondaryReconParams(&reconParams, &img.params);
    
    readBinaryFNames(pathNames.masterFile, pathNames.plainParamsFile, &pathNames);

    readSinoParams(pathNames.masterFile, pathNames.plainParamsFile, &sino.params);
    
    if (reconParams.verbosity>=1)
        printReconParams(&reconParams);
    
    if (reconParams.verbosity>=1)
        printPathNames(&pathNames);
    
    if (reconParams.verbosity>=1)
        printSinoParams(&sino.params);
    
    if (reconParams.verbosity>=1)
        printImgParams(&img.params);

    /**
     *      Parallel stuff
     */
    omp_set_num_threads(reconParams.numThreads);

    /****************************************************************************************************************
     *      Shared 
     ****************************************************************************************************************/
    if (isNeed_SysMatrix && !isMode_sys)
    {
        readSysMatrix(pathNames.sysMatrix, &sino.params, &img.params, &A);
        isExistSysMatrix = 1;
    }

    if (isNeed_sino_DOT_vox)
    {
        sino.vox = (float***) allocateSinoData3DCone(&sino.params, sizeof(float));
        readSinoData3DCone(pathNames.sino, (void***)sino.vox, &sino.params, "float");
        exist_sino_DOT_vox = 1;
    }

    if (isNeed_sino_DOT_wgt)
    {
        sino.wgt = (float***) allocateSinoData3DCone(&sino.params, sizeof(float));
        readSinoData3DCone(pathNames.wght, (void***)sino.wgt, &sino.params, "float");
        exist_sino_DOT_wgt = 1;
    }

    /****************************************************************************************************************
     *      System Matrix
     ****************************************************************************************************************/
    if (isMode_sys)
    {
        readAndAllocateViewAngleList(pathNames.masterFile, pathNames.plainParamsFile, &viewAngleList, &sino.params);

        computeSysMatrix(&sino.params, &img.params, &A, &reconParams, &viewAngleList);
        isExistSysMatrix = 1;
        
        printSysMatrixParams(&A);
        writeSysMatrix(pathNames.sysMatrix, &sino.params, &img.params, &A);

        freeViewAngleList(&viewAngleList);
    }

    /****************************************************************************************************************
     *      proj
     ****************************************************************************************************************/
    if (isMode_proj)
    {
        printf("Forward Projecting ...\n");
        /**
         *      Allocate
         */
        img.projInput = (float***) allocateImageData3DCone( &img.params, sizeof(float), 0);
        sino.projOutput = (float***) allocateSinoData3DCone(&sino.params, sizeof(float));

        /**
         *      Compute
         */
        readImageData3DCone(pathNames.projInput, (void***)img.projInput, &img.params, 0, "float");
        setFloatArray2Value(&sino.projOutput[0][0][0], sino.params.N_beta*sino.params.N_dv*sino.params.N_dw, 0);

        forwardProject3DCone(sino.projOutput, img.projInput, &img.params, &A, &sino.params);

        writeSinoData3DCone(pathNames.projOutput, (void***)sino.projOutput, &sino.params, "float");

        /**
         *      Free
         */
        mem_free_3D((void***)img.projInput);
        mem_free_3D((void***)sino.projOutput);
    }

    /****************************************************************************************************************
     *      like back proj
     ****************************************************************************************************************/
    if (isMode_backprojlike)
    {
        printf("Back Projecting ...\n");
        /**
         *      Allocate
         */
        sino.backprojlikeInput = (float***) allocateSinoData3DCone(&sino.params, sizeof(float));
        img.backprojlikeOutput = (float***) allocateImageData3DCone( &img.params, sizeof(float), 0);


        /**
         *      Compute
         */
        readSinoData3DCone(pathNames.backprojlikeInput, (void***)sino.backprojlikeInput, &sino.params, "float");
        setFloatArray2Value(&img.backprojlikeOutput[0][0][0], img.params.N_x*img.params.N_y*img.params.N_z, 0);

        backProjectlike3DCone(img.backprojlikeOutput, sino.backprojlikeInput, &img.params, &A, &sino.params, &reconParams);
        applyMask(img.backprojlikeOutput, img.params.N_x, img.params.N_y, img.params.N_z);

        writeImageData3DCone(pathNames.backprojlikeOutput, (void***)img.backprojlikeOutput, &img.params, 0, "float");

        /**
         *      Free
         */
        mem_free_3D((void***)sino.backprojlikeInput);
        mem_free_3D((void***)img.backprojlikeOutput);
    }

    /****************************************************************************************************************
     *      wghtRecon
     ****************************************************************************************************************/
    if (isMode_wghtRecon)
    {
        printf("Compute Weight Recon Error ...\n");
        img.wghtRecon = (float***) allocateImageData3DCone( &img.params, sizeof(float), 0);

        initializeWghtRecon(&A, &sino, &img, &reconParams);

        writeImageData3DCone(pathNames.wghtRecon, (void***)img.wghtRecon, &img.params, 0, "float");

        mem_free_3D((void***)img.wghtRecon);
    }
    /****************************************************************************************************************
     *      Initialize
     ****************************************************************************************************************/
    if (isMode_init)
    {


        printf("Initializing Error ...\n");
        /**
         *      Allocate
         */
        sino.estimateSino = (float***) allocateSinoData3DCone(&sino.params, sizeof(float));
        sino.e = (float***) allocateSinoData3DCone(&sino.params, sizeof(float));
        img.vox = (float***) allocateImageData3DCone( &img.params, sizeof(float), 0);
        img.vox_roi = (float***) allocateImageData3DCone( &img.params, sizeof(float), 1);
        img.proxMapInput = (float***) allocateImageData3DCone( &img.params, sizeof(float), 0);
        img.lastChange = (float***) mem_alloc_3D(img.params.N_x, img.params.N_y, reconParams.numZiplines, sizeof(float));
        img.timeToChange = (unsigned char***) mem_alloc_3D(img.params.N_x, img.params.N_y, reconParams.numZiplines, sizeof(unsigned char));
        

        /**
         *      Initialize
         */
        if(strcmp(reconParams.initReconMode, "constant")==0)
        {
            setFloatArray2Value(&img.vox[0][0][0], img.params.N_x*img.params.N_y*img.params.N_z, reconParams.InitVal_recon);
        }
        else if(strcmp(reconParams.initReconMode, "recon")==0)
        {
            readImageData3DCone(pathNames.recon, (void***)img.vox, &img.params, 0, "float");
        }
        else if(strcmp(reconParams.initReconMode, "FDK")==0)
        {
            readImageData3DCone(pathNames.recon, (void***)img.vox, &img.params, 0, "float");
        }
        else
        {
            printf("ERROR: initReconMode '%s' unknown\n", mode);
            exit(-1);
        }
        applyMask(img.vox, img.params.N_x, img.params.N_y, img.params.N_z);

        copyImage2ROI(&img);
        
        setFloatArray2Value(&img.proxMapInput[0][0][0], img.params.N_x*img.params.N_y*img.params.N_z, 0.0);
        setFloatArray2Value(&img.lastChange[0][0][0], img.params.N_x*img.params.N_y*reconParams.numZiplines, 0.0);
        setUCharArray2Value(&img.timeToChange[0][0][0], img.params.N_x*img.params.N_y*reconParams.numZiplines, 0);



        /**
         *      Error Initialization: e = y - Ax
         */
        forwardProject3DCone( sino.estimateSino, img.vox, &img.params, &A, &sino.params);
        /*      e = 1.0 * y + (-1.0) * Ax       */
        floatArray_z_equals_aX_plus_bY(&sino.e[0][0][0], 1.0, &sino.vox[0][0][0], -1.0, &sino.estimateSino[0][0][0], sino.params.N_beta*sino.params.N_dv*sino.params.N_dw);


        
        /**
         *      Write
         */
        
        writeImageData3DCone(pathNames.recon, (void***)img.vox, &img.params, 0, "float");
        writeImageData3DCone(pathNames.reconROI, (void***)img.vox_roi, &img.params, 1, "float");
        writeImageData3DCone(pathNames.proxMapInput, (void***)img.proxMapInput, &img.params, 0, "float");
        write3DData(pathNames.lastChange, (void***)img.lastChange, img.params.N_x, img.params.N_y, reconParams.numZiplines, "float");
        write3DData(pathNames.timeToChange, (void***)img.timeToChange, img.params.N_x, img.params.N_y, reconParams.numZiplines, "unsigned char");

        writeSinoData3DCone(pathNames.errSino, (void***)sino.e, &sino.params, "float");
        writeSinoData3DCone(pathNames.estimateSino, (void***)sino.estimateSino, &sino.params, "float");
        
        /**
         *      Free
         */
        mem_free_3D((void***)sino.estimateSino);
        mem_free_3D((void***)sino.e);

        mem_free_3D((void***)img.vox);
        mem_free_3D((void***)img.vox_roi);
        mem_free_3D((void***)img.proxMapInput);
        mem_free_3D((void***)img.lastChange);
        mem_free_3D((void***)img.timeToChange);

    }
    /****************************************************************************************************************
     *      Reconstruct
     ****************************************************************************************************************/
    if (isMode_recon)
    {
        printf("Reconstructing ...\n");

        /**
         *      Allocate space for sinogram
         */
        sino.estimateSino = (float***) allocateSinoData3DCone(&sino.params, sizeof(float));
        sino.e = (float***) allocateSinoData3DCone(&sino.params, sizeof(float));

        /**
         *      Allocate space for image
         */
        img.vox = (float***) allocateImageData3DCone( &img.params, sizeof(float), 0);
        img.vox_roi = (float***) allocateImageData3DCone( &img.params, sizeof(float), 1);
        img.wghtRecon = (float***) allocateImageData3DCone( &img.params, sizeof(float), 0);
        img.proxMapInput = (float***) allocateImageData3DCone( &img.params, sizeof(float), 0);
        img.lastChange = (float***) mem_alloc_3D(img.params.N_x, img.params.N_y, reconParams.numZiplines, sizeof(float));
        img.timeToChange = (unsigned char***) mem_alloc_3D(img.params.N_x, img.params.N_y, reconParams.numZiplines, sizeof(unsigned char));
        if(reconParams.isPhantomReconReference)
            img.phantom = (float***) allocateImageData3DCone( &img.params, sizeof(float), 0);

        /*
         *      Read sinogram data
         */
        readSinoData3DCone(pathNames.errSino, (void***)sino.e, &sino.params, "float");
        
        /**
         *      Image Initialization
         */
        readImageData3DCone(pathNames.recon, (void***)img.vox, &img.params, 0, "float");
        readImageData3DCone(pathNames.wghtRecon, (void***)img.wghtRecon, &img.params, 0, "float");
        readImageData3DCone(pathNames.proxMapInput, (void***)img.proxMapInput, &img.params, 0, "float");
        read3DData(pathNames.lastChange, (void***)img.lastChange, img.params.N_x, img.params.N_y, reconParams.numZiplines, "float");
        read3DData(pathNames.timeToChange, (void***)img.timeToChange, img.params.N_x, img.params.N_y, reconParams.numZiplines, "unsigned char");

        if(reconParams.isPhantomReconReference)
            readImageData3DCone(pathNames.phantom, (void***)img.phantom, &img.params, 0, "float");

        /**
         *      Reconstruction
         */
        MBIR3DCone(&img, &sino, &reconParams, &A, &pathNames);

        /**
         *      Free allocated data
         */
        mem_free_3D((void***)img.vox);
        mem_free_3D((void***)img.proxMapInput);
        mem_free_3D((void***)img.vox_roi);
        mem_free_3D((void***)img.lastChange);
        mem_free_3D((void***)img.timeToChange);
        if(reconParams.isPhantomReconReference)
            mem_free_3D((void***)img.phantom);

        mem_free_3D((void***)sino.e);
        mem_free_3D((void***)sino.estimateSino);


    }

    /****************************************************************************************************************
     *      Free
     ****************************************************************************************************************/

    if (isExistSysMatrix)
    {
        freeSysMatrix(&A);
    }
    if (exist_sino_DOT_vox)
    {
        mem_free_3D((void***)sino.vox);
    }
    if (exist_sino_DOT_wgt)
    {
        mem_free_3D((void***)sino.wgt);
    }

    



    return 0;
}
