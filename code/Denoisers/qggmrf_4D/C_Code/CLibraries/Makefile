CC=icc
CFLAGS=-Wall -pedantic -fopenmp -O3

OBJECTS=io4d.o MBIRModularUtilities.o allocate.o
SOURCES=io4d.c MBIRModularUtilities.c allocate.c

libOBJECTS=../../../../plainParams/plainParams.o
libSOURCES=../../../../plainParams/plainParams.c



all: ${OBJECTS}

clean:
	rm ${OBJECTS}

${OBJECTS}: ${libOBJECTS}
	${CC} ${CFLAGS}  -c ${SOURCES}

# ${libOBJECTS}: ${libSOURCES}
	


# # To notice when there are changes in the header files
# DEPENDENCIES: .DEPENDENCIES

# .DEPENDENCIES: ${SOURCES}
# 	rm -f ./.DEPENDENCIES
# 	${CC} ${CFLAGS} -MM $^>>./.DEPENDENCIES;

# include .DEPENDENCIES