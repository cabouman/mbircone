CC=icc
CFLAGS=-Wall -pedantic -fopenmp -O3


#Libobjects have been compiled using a different makefile
libOBJECTS=../CLibraries/io4d.o  ../CLibraries/MBIRModularUtilities.o ../CLibraries/allocate.o ../../../../plainParams/plainParams.o
libSOURCES=../CLibraries/io4d.c  ../CLibraries/MBIRModularUtilities.c ../CLibraries/allocate.c ../../../../plainParams/plainParams.c


OBJECTS=icd4d.o recon4D.o go_reconstruct.o
SOURCES=icd4d.c recon4D.c go_reconstruct.c


executable=go_reconstruct

all: ${executable}

clean:
	rm ${OBJECTS} ${executable}

${executable}:${OBJECTS} ${libOBJECTS}
	${CC} ${CFLAGS} ${libOBJECTS}  ${OBJECTS} -o ${executable}

${OBJECTS}: ${libOBJECTS}
	${CC} ${CFLAGS} -c ${SOURCES}

${libOBJECTS}: ${libSOURCES}



