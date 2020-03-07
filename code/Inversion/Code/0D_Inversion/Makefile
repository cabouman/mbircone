CC=icc
CFLAGS=-Wall -pedantic   -O3 -fopenmp


#Libobjects have been compiled using a different makefile
libOBJECTS=../0A_CLibraries/io3d.o  ../0A_CLibraries/MBIRModularUtilities3D.o ../0A_CLibraries/allocate.o ../../../plainParams/plainParams.o
libSOURCES=../0A_CLibraries/io3d.c  ../0A_CLibraries/MBIRModularUtilities3D.c ../0A_CLibraries/allocate.c ../../../plainParams/plainParams.c


OBJECTS=icd3d.o recon3DCone.o go_reconstruct.o
SOURCES=icd3d.c recon3DCone.c go_reconstruct.c


executable=go_reconstruct

all: ${executable}

clean:
	rm ${OBJECTS} ${executable}

${executable}:${OBJECTS} ${libOBJECTS}
	${CC} ${CFLAGS} ${libOBJECTS} ${OBJECTS} -o ${executable}

${OBJECTS}: ${libOBJECTS}
	${CC} ${CFLAGS} -c ${SOURCES}

${libOBJECTS}: ${libSOURCES}



